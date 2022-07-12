import numpy as np
import pandas as pd

from astropy.stats import sigma_clip
from astropy.timeseries import LombScargle
from astropy.table import Table, Column, MaskedColumn, join, vstack, unique

from scipy.interpolate import interp1d, LSQUnivariateSpline, UnivariateSpline
from scipy.signal import medfilt, find_peaks, savgol_filter, correlate
from scipy.stats import chisquare, rv_histogram, shapiro, anderson, jarque_bera, norm
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning

from gatspy.periodic import LombScargleFast

from bisect import bisect

import os
import warnings

import lightkurve


###################################
#                                 #
# Kepler Short Cadence Detrending #
#                                 #
###################################

def determine_window(lc_table):
    """
    Do a Lomb-Scargle and decide the window.
    Assumes columns time and flux.
    Does not have any thresholding, but returns FAP value.
    """

    # Dealing with masked values if any
    if isinstance(lc_table['flux'],MaskedColumn):
        fluxes = lc_table['flux'][~lc_table['flux'].mask]
        times = lc_table['time'][~lc_table['flux'].mask].jd
    else:
        fluxes = lc_table['flux']
        times = lc_table['time'].jd

    lomb = LombScargle(times, fluxes)
    frequency = np.linspace(.5,5,2000)
    power = lomb.power(frequency)

    fap = lomb.false_alarm_probability(power.max())

    fmax = frequency[np.argmax(power)]

    if fmax < 1:
        window = 401
    elif fmax >= 3:
        window = 101
    else:
        window = 201

    return window, fmax, fap.value
        

def detrend_lc(lc_table, clip_sigma=3.5, window=401, clean=False):
    """
    Given a light curve table, with columns "time" and "flux"
    add new columns median_trend flux_detrended containing the median trend and
    detrended light curve.
    """
    
    if clean and ('median_trend' in lc_table.colnames): # redoing a trend, need to remove previous trend
        lc_table.remove_columns(['median_trend', 'flux_detrended'])
        

    # Sigma clipping
    with warnings.catch_warnings():  # Ignore warnings due to NaNs or Infs
        warnings.simplefilter("ignore")
        clipped_flux = sigma_clip(lc_table["flux"], sigma=clip_sigma)

    # Interpolating the masked values for detrending purposes
    interp_func = interp1d(lc_table["time"].jd[~clipped_flux.mask],
                           clipped_flux.data[~clipped_flux.mask],
                           fill_value="extrapolate")
    clipped_flux = interp_func(lc_table["time"].jd)

    # Get the median trend
    median_trend = medfilt(clipped_flux,window)

    # Detrend light curve
    detrended_flux = (lc_table["flux"] - median_trend)/median_trend
    detrended_flux.name = "flux_detrended"

    # Putting it all in the table
    lc_table.add_column(Column(name="median_trend", data=median_trend))
    lc_table.add_column(detrended_flux)
    

def find_nearest(arr, val):

    diff_arr = np.abs(arr - val)
    return np.where(diff_arr == diff_arr.min())[0][0]


def get_fit_stats(lc_table):
    """
    Takes in fitted light curve, assumes column names as produced by
    detrend_lc and performed a number of statistical tests.
    """

    # Dealing with masked values if any
    if isinstance(lc_table['flux_detrended'],MaskedColumn):
        det_fluxes = lc_table['flux_detrended'][~lc_table['flux_detrended'].mask]
    else:
        det_fluxes = lc_table['flux_detrended']

    # removing any infs (should be rare)
    det_fluxes = det_fluxes[~np.isinf(det_fluxes)]
    
    nbins = 1000
    hist,bin_edges = np.histogram(det_fluxes, bins=nbins)
    bin_mids = (bin_edges[:-1]+bin_edges[1:])/2

    # Putting all the tests in a dictionary
    stats_dict = {}

    # Need the cumulative distribution to find the percentiles
    hist_cumudist = np.cumsum(hist)
    tot_vals = hist_cumudist[-1]

    # 90% width
    perc_5 = bin_mids[bisect(hist_cumudist, tot_vals*0.05)]
    perc_95 = bin_mids[bisect(hist_cumudist, tot_vals*0.95)]
    stats_dict["90% width"] = perc_95 -perc_5     

    # jarque-bera test
    stats_dict["jarque-bera"] = jarque_bera(hist)[0] # Just getting the stat not the p-value

    # Chi-squared test on the histogram
    low_idx = bisect(hist_cumudist, tot_vals*0.34)
    high_idx = bisect(hist_cumudist, tot_vals*0.66)+1

    sigma = bin_mids[high_idx] - bin_mids[low_idx]
    mu = 0
    
    normalization = np.max(hist)/norm.pdf(0, mu, sigma)
    gaussian_fit = norm.pdf(bin_mids, mu, sigma)*normalization

    low_idx = bisect(hist_cumudist, tot_vals*0.17)
    high_idx = bisect(hist_cumudist, tot_vals*0.83)+1
    stats_dict["chisq hist"] = chisquare(hist[low_idx:high_idx],
                                         gaussian_fit[low_idx:high_idx])[0] # Just getting the stat not the p-value

    # Looking for peaks (first smooth)
    mid_ind = find_nearest(bin_mids,0)
    
    high_idx = np.where(hist[mid_ind:]==0)[0]
    if not len(high_idx):
        high_idx = len(hist)-1
    else:
        high_idx = high_idx[0] + mid_ind
        
    low_idx = np.where(hist[:mid_ind]==0)[0]
    if not len(low_idx):
        low_idx = 0
    else:
        low_idx = low_idx[-1]
    peaks,_ = find_peaks(medfilt(hist,51)[low_idx:high_idx])

    # Count peaks
    stats_dict["num peaks"] = len(peaks)
    
    return stats_dict


def score_tests(stats_dict):
    """
    Scores the results of the statistical tests.
    Expects a dictionary with the keys: 
        90% width (> 0.02 = fail)
        jarque-bera (> 20000 = fail)
        chisq hist(> 5000 = fail)
        num peaks (> 1 = fail)

    Score >= 2 is considered failure.
    """
    score = 0

    if stats_dict["90% width"] > 0.02:
        score += 1
    if stats_dict["jarque-bera"] > 20000:
        score += 1
    if stats_dict["chisq hist"] > 5000:
        score += 1
    if stats_dict["num peaks"] > 1:
        score += 1

    return score


def split_on_gaps(lc_table, min_part_num, max_gap=1):
    """
    Given a light curve table split it into seperate tables any time there is a gap of
    more than max_gap hours. Also adds a column "partition" that starts at min_part_num
    and goes up with each new table.

    Returns the list of tables.
    """
    
    maxgap_jd = max_gap/24
    gap_locs = np.where((lc_table['time'].jd[1:]-lc_table['time'].jd[:-1]) > maxgap_jd)[0]
    
    lc_table_list = []
    beg_ind = 0
    part_num = min_part_num
    for loc in gap_locs:
        part_table = lc_table[beg_ind:loc+1]
        part_table["partition"] = part_num
        lc_table_list.append(part_table)
        
        beg_ind = loc + 1
        part_num += 1
    
    last_part = lc_table[beg_ind:]
    last_part["partition"] = part_num
    lc_table_list.append(last_part)
        
    return lc_table_list


def detrend_and_score(kid, quarters, clip_sigma=3.5):
    """
    Given a kepler ID and list of quarters, uses lightKurve do download all 
    data, detrend, combine into a single table, and score the detrend fit.

    Returns
    -------
    response : astropy.table.Table
        Detrended light curve table. With the detrending window, score, and 
        goodness of fit test results in the metadata.
    """

    # Getting all the light curves
    lc_filelist = lightkurve.search_lightcurvefile(f"KIC {kid}", cadence='short', quarter=quarters).download_all()

    lc_table_list = []
    for lc_file in lc_filelist:

        min_part_num = len(lc_table_list)
        lc_tables = split_on_gaps(lc_file.SAP_FLUX.to_table(),min_part_num)
        lc_table_list += lc_tables

    # Detrending
    window = None
    for lc_table in lc_table_list:

        if not window: # Just take the first window caluclation
            window,_,_ = determine_window(lc_table)

        detrend_lc(lc_table, clip_sigma=clip_sigma, window=window, clean=True)

    # Combining into a single light curve
    lc_table = vstack(lc_table_list)
    lc_table.meta["detrending window"] = window # record the detrending window

    # Goodness-of-fit testing
    stats_dict = get_fit_stats(lc_table)
    score = score_tests(stats_dict)
    
    lc_table.meta.update(stats_dict)
    lc_table.meta["score"] = score

    return lc_table


##################################################################
# Detrending from Davenport 2016                                 #
#                                                                #
# https://github.com/jradavenport/appaloosa                      #
# https://ui.adsabs.harvard.edu/abs/2016ApJ...829...23D/abstract #
#                                                                #
##################################################################

def multi_boxcar(time, flux, error, numpass=3, kernel=2.0,
                 sigclip=5, pcentclip=5, returnindx=False,
                 debug=False):
    '''
    Boxcar smoothing with multi-pass outlier rejection. Uses both errors
    and local scatter for rejection Uses Pandas rolling median filter.

    Parameters
    time : 1-d numpy array
    flux : 1-d numpy array
    error : 1-d numpy array
    numpass : int, optional
        the number of passes to make over the data. (Default = 3)
    kernel : float, optional
        the boxcar size in hours. (Default is 2.0)
        Note: using whole numbers is probably wise here.
    sigclip : int, optional
        Number of times the standard deviation to clip points at
        (Default is 5)
    pcentclip : int, optional
        % of data to clip for outliers, i.e. 5= keep 5th-95th percentile
        (Default is 5)
    debug : bool, optional
        used to print out troubleshooting things (default=False)

    Returns
    -------
    The smoothed light curve model
    '''

    # the data within each gap range

    # This is annoying: https://pandas.pydata.org/pandas-docs/stable/gotchas.html#byte-ordering-issues
    #flux = flux.byteswap().newbyteorder()
    #time = time.byteswap().newbyteorder()
    #error = error.byteswap().newbyteorder()

    flux_i = pd.DataFrame({'flux':flux,'error_i':error,'time_i':time})
    time_i = np.array(time)
    error_i = error
    indx_i = np.arange(len(time)) # for tracking final indx used
    exptime = np.nanmedian(time_i[1:]-time_i[:-1])

    nptsmooth = int(kernel/24.0 / exptime)
    if debug is True:
        print('# of smoothing points: '+str(nptsmooth))
        print(kernel, exptime)
    
    if (nptsmooth < 4):
        nptsmooth = 4

    if nptsmooth >= len(flux_i):
        nptsmooth = len(flux_i)//2

    if debug is True:
        print('# of smoothing points: '+str(nptsmooth))

    # now take N passes of rejection on it
    for k in range(0, numpass):
        
        # rolling median in this data span with the kernel size
        flux_i['flux_i_sm'] = flux_i.flux.rolling(nptsmooth, center=True, min_periods=4).median()
        flux_i = flux_i.dropna(how='any')

        if (flux_i.shape[0] > 1):
            flux_i['diff_k'] = flux_i.flux-flux_i.flux_i_sm
            lims = np.nanpercentile(flux_i.diff_k, (pcentclip, 100-pcentclip))

            # iteratively reject points
            # keep points within sigclip (for phot errors), or
            # within percentile clip (for scatter)
            ok = np.logical_or((np.abs(flux_i.diff_k / flux_i.error_i) < sigclip),
                               (lims[0] < flux_i.diff_k) & (flux_i.diff_k < lims[1]))
            if debug is True:
                print('k = '+str(k))
                print('number of accepted points:',sum(ok))

            flux_i = flux_i[ok]

    flux_sm = np.interp(time, flux_i.time_i, flux_i.flux)

    indx_out = flux_i.index.values

    if returnindx is False:
        return flux_sm
    else:
        return np.array(indx_out, dtype='int')

    
def _sinfunc(t, per, amp, t0, yoff):
    '''
    Simple function defining a single Sine curve for use in curve_fit applications
    Defined as:
        F = sin( (t - t0) * 2 pi / period ) * amplitude + offset

    Parameters
    ----------
    t : 1-d numpy array
        array of times
    per : float
        sin period
    amp : float
        amplitude
    t0 : float
        phase zero-point
    yoff : float
        linear offset

    Returns
    -------
    F, array of fluxes defined by sine function
    '''

    return np.sin((t - t0) * 2.0 * np.pi / per) * amp  + yoff

def _sinfunc2(t, per1, amp1, t01, per2, amp2, t02, yoff):
    '''
    Simple function defining two Sine curves for use in curve_fit applications
    Defined as:
        F = sin( (t - t0_1) * 2 pi / period_1 ) * amplitude_1 + \
            sin( (t - t0_2) * 2 pi / period_2 ) * amplitude_2 + offset

    Parameters
    ----------
    t : 1-d numpy array
        array of times
    per1 : float
        sin period 1
    amp1 : float
        amplitude 1
    t01 : float
        phase zero-point 1
    per2 : float
        sin period 2
    amp2 : float
        amplitude 2
    t02 : float
        phase zero-point 2
    yoff : float
        linear offset

    Returns
    -------
    F, array of fluxes defined by sine function
    '''

    output = np.sin((t - t01) * 2.0 * np.pi / per1) * amp1 + \
             np.sin((t - t02) * 2.0 * np.pi / per2) * amp2 + yoff
    return output


def fit_sine(time, flux, error, maxnum=5, nper=20000,
             minper=0.1, maxper=30.0, plim=0.25,
             per2=False, returnmodel=True, debug=False):
    '''
    Use Lomb Scargle to find a periodic signal. If it is significant then fit
    a sine curve and subtract. Repeat this procedure until no more periodic
    signals are found, or until maximum number of iterations has been reached.

    Note: this is where major issues were found in the light curve fitting as
    of Davenport (2016), where the iterative fitting was not adequately
    subtracting "pointy" features, such as RR Lyr or EBs. Upgrades to the
    fitting step are needed! Or, don't use iterative sine fitting...

    Idea for future: if L-S returns a significant P, use a median fit of the
    phase-folded data at that P instead of a sine fit...

    Parameters
    ----------
    time : 1-d numpy array
    flux : 1-d numpy array
    error : 1-d numpy array
    maxnum : int, optional
        maximum number of iterations to try finding periods at
        (default=5)
    nper : int, optional
        number of periods to search over with Lomb Scargle
        (defeault=20000)
    minper : float, optional
        minimum period (in units of time array, nominally days) to search
        for periods over (default=0.1)
    maxper : float, optional
        maximum period (in units of time array, nominally days) to search
        for periods over (default=30.0)
    plim : float, optional
        Lomb-Scargle power threshold needed to define a "significant" period
        (default=0.25)
    per2 : bool, optional
        if True, use the 2-sine model fit at each period. if False, use normal
        1-sine model (default=False)
    returnmodel : bool, optional
        if True, return the combined sine model. If False, return the
        data - model (default=True)
    debug : bool, optional
        used to print out troubleshooting things (default=False)

    Returns
    -------
    If returnmodel=True, output = combined sine model (default=True)
    If returnmodel=False, output = (data - model)
    '''

    flux_out = np.array(flux, copy=True)
    sin_out = np.zeros_like(flux) # return the sin function!

    # total baseline of time window
    dt = np.nanmax(time) - np.nanmin(time)

    medflux = np.nanmedian(flux)

    for k in range(0, maxnum):
        # Use Jake Vanderplas faster version!
        pgram = LombScargleFast(fit_offset=False)
        pgram.optimizer.set(period_range=(minper,maxper))
        pgram = pgram.fit(time, flux_out - medflux, error)

        df = (1./minper - 1./maxper) / nper
        f0 = 1./maxper
        pwr = pgram.score_frequency_grid(f0, df, nper)

        freq = f0 + df * np.arange(nper)
        per = 1./freq

        pok = np.where((per < dt) & (per > minper))
        pk = per[pok][np.argmax(pwr[pok])]
        pp = np.max(pwr)

        if debug is True:
            print('trial (k): '+str(k)+'.  peak period (pk):'+str(pk)+
                  '.  peak power (pp):'+str(pp))

        # if a period w/ enough power is detected
        #warnings.filterwarnings("error")
        if (pp > plim):
            # fit sin curve to window and subtract
            if per2 is True:
                p0 = [pk, 3.0 * np.nanstd(flux_out-medflux), 0.0,
                      pk/2., 1.5 * np.nanstd(flux_out-medflux), 0.1, 0.0]
                try:
                    pfit, pcov = curve_fit(_sinfunc2, time, flux_out-medflux, p0=p0)
                    if debug is True:
                        print('>>', pfit)
                        #print(pcov)
                except RuntimeError:
                    pfit = [pk, 0., 0., 0., 0., 0., 0.]
                    if debug is True:
                        print('Curve_Fit2 no good')
                #except OptimizeWarning as ow:
                #    print(ow)
                    #pfit = [pk, 0., 0., 0.]
                    #print(pfit)
                    #print(pcov)

                flux_out = flux_out - _sinfunc2(time, *pfit)
                sin_out = sin_out + _sinfunc2(time, *pfit)

            else:
                p0 = [pk, 3.0 * np.nanstd(flux_out-medflux), 0.0, 0.0]
                try:
                    pfit, pcov = curve_fit(_sinfunc, time, flux_out-medflux, p0=p0)
                    if debug:
                        print(pfit)
                        #print(pcov)
                except RuntimeError:
                    pfit = [pk, 0., 0., 0.]
                    if debug is True:
                        print('Curve_Fit no good')
                #except OptimizeWarning as ow:
                #    print(ow)
                    #pfit = [pk, 0., 0., 0.]
                    #print(pfit)
                    #print(pcov)

                flux_out = flux_out - _sinfunc(time, *pfit)
                sin_out = sin_out + _sinfunc(time, *pfit)
        #warnings.resetwarnings()

        # add the median flux for this window BACK in
        sin_out = sin_out + medflux

    if returnmodel is True:
        return sin_out
    else:
        return flux_out

    
def irls_spline(time, flux, error, Q=400.0, ksep=0.07, numpass=5, order=3, debug=False):
    '''
    IRLS = Iterative Re-weight Least Squares
    Do a multi-pass, weighted spline fit, with iterative down-weighting of
    outliers. This is a simple, highly flexible approach. Suspiciously good
    at times...

    Originally described by DFM: https://github.com/dfm/untrendy
    Likley not adequately reproduced here.

    uses scipy.interpolate.LSQUnivariateSpline

    Parameters
    ----------
    time : 1-d numpy array
    flux : 1-d numpy array
    error : 1-d numpy array
    Q : float, optional
        the penalty factor to give outlier data in subsequent passes
        (deafult is 400.0)
    ksep : float, optional
        the spline knot separation, in units of the light curve time
        (default is 0.07)
    numpass : int, optional
        the number of passes to take over the data (default is 5)
    order : int, optional
        the spline order to use (default is 3)
    debug : bool, optional
        used to print out troubleshooting things (default=False)

    Returns
    -------
    the final spline model
    '''

    weight = 1. / (error**2.0)

    knots = np.arange(np.nanmin(time) + ksep, np.nanmax(time) - ksep, ksep)

    if debug is True:
        print('IRLSSpline: knots: ', np.shape(knots))
        print('IRLSSpline: time: ', np.shape(time), np.nanmin(time), time[0], np.nanmax(time), time[-1])
        print('IRLSSpline: <weight> = ', np.mean(weight))
        print(np.where((time[1:] - time[:-1] < 0))[0])

    for k in range(numpass):
        try:
            spl = LSQUnivariateSpline(time, flux, knots, w=weight, k=order, check_finite=True)
        except:
            print("Fell through to UnivariateSpline")
            spl = UnivariateSpline(time, flux, w=weight, k=order, s=1)

        chisq = ((flux - spl(time))**2.) / (error**2.0)

        weight = Q / ((error**2.0) * (chisq + Q))

    return spl(time)


def model_lightcurve(time, flux, error, debug=False):

    '''
    Construct a model light curve. (mode=davenport detrending from appaloosa)

    Parameters:
    ------------
    time : numpy array
    flux : numpy array
    errors : numpy array
    debug : bool 
        Setting to true will print debugging output

    Returns
    -------
    response : numpy array
        The model light curve
    '''

    time = np.array(time)
    flux = np.array(flux)
    error = np.array(error)

    # do iterative rejection and spline fit - like FBEYE did
    # also like DFM & Hogg suggest w/ BART
    box1 = multi_boxcar(time, flux, error, kernel=2.0, numpass=2)
    box1 = multi_boxcar(time, flux, error, kernel=60, numpass=2)

    sin1 = fit_sine(time, box1, error, maxnum=5, maxper=(max(time)-min(time)), per2=False, debug=debug)
    box3 = multi_boxcar(time, flux - sin1, error, kernel=0.3, debug=debug)
    box3 = multi_boxcar(time, flux - sin1, error, kernel=9, debug=debug)
    t = np.array(time)
    dt = np.nanmedian(t[1:] - t[0:-1])
    exptime_m = (np.nanmax(time) - np.nanmin(time)) / len(time)

    # ksep used to = 0.07...
    flux_model_i = irls_spline(time, box3, error, numpass=20, ksep=exptime_m*10., debug=debug)
                                          
    flux_model_i += sin1 

    return flux_model_i


#######################################
#                                     #
#   Kepler Long Cadence Detrending    #
#                                     #
# (uses Devenport detrending and some #
#  functions from short cadence bit)  # 
#                                     #
#######################################

def detrend_longcadence_lc(lc_table, debug=False):
    """
    Given a light curve table, with columns "time" and "flux"
    add new columns flux_model flux_detrended containing the model quiescent
    flux (using Davenport's method) and detrended light curve.

    Also removes all masked values from the table.
    """

    # Removingf previous trend if needed
    if 'flux_model' in lc_table.colnames:
        lc_table.remove_columns('flux_model')
    if 'flux_detrended' in lc_table.colnames:
        lc_table.remove_columns('flux_detrended')

    # Dealing with masked values if any
    if isinstance(lc_table['flux'], MaskedColumn):
        lc_table.remove_rows(lc_table['flux'].mask)

    # Get the flux model (davenport method)
    flux_model = model_lightcurve(lc_table["time"].jd, lc_table["flux"], lc_table["flux_err"], debug=debug)  

    # Detrend light curve
    detrended_flux = (lc_table["flux"] - flux_model)/flux_model
    detrended_flux.name = "flux_detrended"

    # Putting it all in the table
    lc_table.add_column(Column(name="flux_model", data=flux_model))
    lc_table.add_column(detrended_flux)


def longcad_detrend_and_score(kid, quarters, debug=False):
    """
    Given a kepler ID and list of quarters, uses lightKurve do download all 
    long cadence data, detrend, combine into a single table, and score 
    the detrend fit.

    Returns
    -------
    response : astropy.table.Table
        Detrended light curve table. With the score and 
        goodness of fit test results in the metadata.
    """

    # Getting all the light curves
    lc_filelist = lightkurve.search_lightcurvefile(f"KIC {kid}", cadence='long', quarter=quarters).download_all()

    lc_table_list = []
    for lc_file in lc_filelist:

        # Sometimes lightkurve returns data for additional KIDs (put in issue 3/27/20)
        if lc_file.targetid != int(kid):
            continue

        min_part_num = len(lc_table_list)
        lc_tables = split_on_gaps(lc_file.SAP_FLUX.to_table(),min_part_num)
        lc_table_list += lc_tables

    # Detrending
    for lc_table in lc_table_list:
        detrend_longcadence_lc(lc_table, debug=debug)

    # Combining into a single light curve
    lc_table = vstack(lc_table_list)

    # Adding in the detrended flux error
    lc_table["flux_det_err"] = lc_table["flux_err"]/lc_table["flux_model"]
    
    # Goodness-of-fit testing
    stats_dict = get_fit_stats(lc_table)
    score = score_tests(stats_dict)
    
    lc_table.meta.update(stats_dict)
    lc_table.meta["score"] = score

    return lc_table


def longcad_redetrend_and_score(lc_table, debug=False):
    """
    For the injected flares, are starting with an already downloaded light curve.
    This also works for redoing the detrending on any light curve.

    Returns
    -------
    response : astropy.table.Table
        Detrended light curve table. With the score and 
        goodness of fit test results in the metadata.
    """

    # Detrending each partition individually
    lc_table_list = []
    for partition in np.unique(lc_table["partition"]):
        temp_lc = lc_table[lc_table["partition"]==partition]
        detrend_longcadence_lc(temp_lc, debug=debug)
        lc_table_list.append(temp_lc)
    
    lc_table = vstack(lc_table_list)


    # Adding in the detrended flux error
    lc_table["flux_det_err"] = lc_table["flux_err"]/lc_table["flux_model"]
    
    # Goodness-of-fit testing
    stats_dict = get_fit_stats(lc_table)
    score = score_tests(stats_dict)
    
    lc_table.meta.update(stats_dict)
    lc_table.meta["score"] = score

    return lc_table



