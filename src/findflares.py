import numpy as np

from astropy.table import Table, Column, MaskedColumn
from astropy.time import Time

from scipy.optimize import curve_fit

from detrend import find_nearest


def double_exponential(x, alpha, beta, a, b, c): 
    """
    Double exponential function.
    """
    return alpha * np.exp(-(x-a)/b) + beta * np.exp(-(x-a)/c)

def gaussian(x, a0, a1, a2):
    """
    Gaussian function as in IDL GAUSSFIT, with 3 parameters.
    """
    z = (x - a1) / a2
    y = a0 * np.exp(-z**2 / 2)
    return y


def find_flare_edges(fluxes, start_ind, end_ind, qui_flux=0):
    """Finding the edges of a detected flare"""

    if isinstance(qui_flux, (int, float)):
        while (fluxes[start_ind] > qui_flux) and (start_ind > 0):
            start_ind -= 1
        
        while (fluxes[end_ind] > qui_flux) and (end_ind < (len(fluxes)-1)):
            end_ind += 1

    else:  # list
        while (fluxes[start_ind] > qui_flux[start_ind]) and (start_ind > 0):
            start_ind -= 1
        
        while (fluxes[end_ind] > qui_flux[end_ind]) and (end_ind < (len(fluxes)-1)):
            end_ind += 1

    return start_ind, end_ind


def phi_vv_stats(lc_table):
    """
    Calculating the phi_vv values and seperating them into candidates, null1/2, and excluded.

    * "ws" referest to Welch and Stetson, the originators of the phi_vv selection criteria

    Note this function also modifies lc_table and returns the new version, this is because it takes
    way longer to modify it in place.
    """

    # Calculating the detrended flux error if necessary
    if "flux_det_err" not in lc_table.colnames:
        lc_table["flux_det_err"] = lc_table["flux_err"]/lc_table["median_trend"]

    # Removing masked points if necessary
    if type(lc_table['flux_detrended']) == MaskedColumn:
        lc_table = lc_table[~lc_table['flux_detrended'].mask]

    # Calculating phi_vv
    Vrel_sigma = lc_table['flux_detrended']/lc_table['flux_det_err']
    phi_vv = Vrel_sigma[:-1]*Vrel_sigma[1:]

    # Seperating the phi_vvs
    Vrel_sign = lc_table['flux_detrended'] > 0

    candidates = Vrel_sign[:-1] & Vrel_sign[1:]
    excluded = (~Vrel_sign[:-1]) & (~Vrel_sign[1:])
    null_1 = (Vrel_sign[:-1] ^ Vrel_sign[1:]) & Vrel_sign[:-1]
    null_2 = (Vrel_sign[:-1] ^ Vrel_sign[1:]) & Vrel_sign[1:]

    # Puttin the info in a table
    ws_stat_table = Table(names=["time_1","time_2","phi_vv",
                                 "candidates","excluded", "null_1", "null_2"],
                          data=[lc_table['time'].jd[:-1],lc_table['time'].jd[1:],
                                phi_vv, candidates, excluded, null_1, null_2])

    return ws_stat_table, lc_table


def get_hist_bins(max_hist_val, max_null_val=None, bin_width=0.73, verbose=True):
    """
    Given a maximum value and bin width, get a list of histogram bin edges.

    For phi_vv calculations, usually we want:
    max_hist_val = int(ws_stat_table['phi_vv'][ws_stat_table['candidates']].max())
    """

    # Setting up the histogram parameters
    if (max_null_val is None) or (max_hist_val < (max_null_val+200*bin_width)):
        nbins = int(max_hist_val/bin_width + 1)
        bins = np.linspace(0, max_hist_val, nbins)
    else:
        nbins = int(max_null_val/bin_width + 101)
        bins = np.concatenate((np.linspace(0, max_null_val+100*bin_width, nbins),
                               np.linspace(max_null_val+101*bin_width, max_hist_val, 100)))

    if verbose:
        print(f"Number of bins: {len(bins)}")

    return bins
    

def fit_phi_vv_histogram(ws_stat_table, bins, verbose=True):
    """
    Performing the double-exponential fit on the null distributions.
    """

    # Making the null distribution histograms
    hist1, bin_edges = np.histogram(-ws_stat_table['phi_vv'][ws_stat_table['null_1']], bins=bins)
    hist2, bin_edges = np.histogram(-ws_stat_table['phi_vv'][ws_stat_table['null_2']], bins=bins)

    # We only need to work with the bins up till the last non-zero entry
    hist1_max = np.where(hist1 > 0)[0].max() + 1
    hist2_max = np.where(hist2 > 0)[0].max() + 1
    max_ind = hist1_max if (hist1_max > hist2_max) else hist2_max

    if verbose:
        print(f"Number of null sample bins: {max_ind}")

    hist1 = hist1[:max_ind]
    hist2 = hist2[:max_ind]
    bin_edges = bin_edges[:max_ind+1]

    # Getting the middle of the bins
    bin_mids = (bin_edges[:-1]+bin_edges[1:])/2

    hist_vals = np.concatenate((hist1, hist2))
    x_vals = np.concatenate((bin_mids, bin_mids))

    # Getting initial guessed for the fitting function (better than the built in defaults)
    half_height = int(hist_vals.max()/2)
    fwhm1 = bin_mids[find_nearest(hist1,half_height)]
    fwhm2 = bin_mids[find_nearest(hist2,half_height)]

    # Fitting the null distributions to the double exponential function
    popt, pcov = curve_fit(double_exponential, x_vals, hist_vals, [half_height,half_height,0,fwhm1,fwhm2])

    if verbose:
        print(f"{popt[0]:.0f}*e^-(x-{popt[2]:.2f})/{popt[3]:.2f} + {popt[1]:.0f}*e^-(x-{popt[2]:.2f})/{popt[4]:.2f}")

    return popt, pcov


def calculate_pvals(ws_stat_table, bins, fit_params, clean=False):
    """
    Calculating the p-values for the candidate phi_vv values.

    This function produces not output, but instead adds a column "pval" to ws_stat_table.
    """

    if clean and ("pval" in ws_stat_table.colnamse): # redoing calculation, remove previous result
        ws_stat_table.remove_column('pval')
    
    # Calculating the p-values
    bin_mids = (bins[:-1] + bins[1:])/2
    fit_hist = double_exponential(bin_mids,*fit_params)
    cumu_dist = np.cumsum(fit_hist)/sum(fit_hist)
    pvals_prime2 = 1 - cumu_dist

    pvals = []
    for cand_val in ws_stat_table['phi_vv'][ws_stat_table['candidates']]:
        pvals.append(pvals_prime2[find_nearest(bin_mids,cand_val)])
    pvals = np.array(pvals)

    # Adding the p-values to ws_stat_table
    ws_stat_table.add_column(Column(name="pval",dtype=float,length=len(ws_stat_table)))

    for row in ws_stat_table:
        if row['candidates']:
            row['pval'] = pvals_prime2[find_nearest(bin_mids,row['phi_vv'])]
        else:
            row['pval'] = np.nan

    
def perform_fdr_analysis(ws_stat_table, alpha=0.1, clean=False, verbose=True):
    """
    Performing the false discovery rate analysis (Miller et al.).

    Alpha is the false discovery rate, default is 10%.

    Expects columns "candidates", "pval", and "phi_vv" in ws_stat_table.

    ws_stat_table will have the column "pass_fdr" added.

    Returns p_val threshold, phi_vv threshold
    """

    if clean and ("pass_fdr" in ws_stat_table.colnamse): # redoing calculation, remove previous result
        ws_stat_table.remove_column('pass_fdr')

        
    candidate_table = ws_stat_table[ws_stat_table['candidates']]
    candidate_table.sort("pval")

    num_pvals = len(candidate_table)
    j_alpha = alpha*np.linspace(1,num_pvals,num_pvals)/num_pvals

    diff = candidate_table["pval"]-j_alpha
    thresh_row = candidate_table[np.where(diff <= 0)[0].max()]

    pval_thresh = thresh_row["pval"]
    thresh = thresh_row["phi_vv"]

    if verbose: 
        print(f"Pval threshold: {pval_thresh:.3}")
        print(f"Phi_vv threshold: {thresh:.3}")

    ws_stat_table.add_column(Column(name="pass_fdr",dtype=bool,length=len(ws_stat_table)))
    ws_stat_table["pass_fdr"][ws_stat_table["candidates"] & (ws_stat_table["phi_vv"] > thresh)] = True

    return pval_thresh, thresh


def apply_sigma_threshold(lc_table, ws_stat_table, threshold=2.5, nbins=1000,
                          clean=False, verbose=True, remove_artifacts=True):
    """
    Threshold is in standard deviations. 
    We want the std of only the quiescent fluxes, so we will remove the points that pass the FDR test, 
    and also the single very negative points that are Kepler artifacts.

    Expects column "flux_detrended" in lc_table.

    Expects column "pass_fdr" in ws_stat_table.

    ws_stat_table will have the column "pass_std" added.
    """

    if clean and ("pass_std" in ws_stat_table.colnames): # redoing calculation, remove previous result
        ws_stat_table.remove_column('pass_std')

    if remove_artifacts:
        # Removing the Kepler artifacts by looking at the histogram of the flux derivative
        # and removing outliers.

        deriv = lc_table["flux_detrended"] - np.concatenate((lc_table["flux_detrended"][1:],lc_table["flux_detrended"][:1]))
        dhist, bin_edges = np.histogram(deriv, bins=nbins)

        bin_mids = (bin_edges[:-1]+bin_edges[1:])/2
        first_zero = bin_mids[int(nbins/2) + np.argwhere(dhist[int(nbins/2):] == 0)[0][0]]

        try:
            parameters, covariance = curve_fit(gaussian, bin_mids, dhist, [dhist.max(),0,first_zero])

            # Rachel used 4 for the multiplier on the std, how to decide this? 
            good_flux_mask = np.abs(deriv) < (parameters[1] + 4*parameters[2])

        except RuntimeError: # just skip if can't fit
            good_flux_mask = np.abs(deriv) >= 0

        if verbose:
            print(f"{sum(~good_flux_mask)}/{len(deriv)} Kepler artifact points.")
    else:
        good_flux_mask = np.ones(len(lc_table), dtype=bool)

    # Getting the std of the quiescent flux
    fail_fdr = ~(np.concatenate((ws_stat_table["pass_fdr"], [True])) | np.concatenate(([True], ws_stat_table["pass_fdr"])))
    qui_flux_mask = good_flux_mask & fail_fdr
    std = np.std(lc_table["flux_detrended"][qui_flux_mask])

    if verbose:
        print(f"Calculated 'quiescent flux' standard deviation: {std}")

    # We require both points involved in a phi_vv value to pass the threshold
    check_1 = (lc_table["flux_detrended"][:-1]/std) > threshold
    check_2 = (lc_table["flux_detrended"][1:]/std) > threshold
    ws_stat_table["pass_std"] = check_1 & check_2 & ws_stat_table['candidates']

    if verbose:
        print(f"Sigma threshold: {std*threshold}")

    return std*threshold

    
def make_flare_list(ws_stat_table, lc_table, padding=10):
    """
    Turn the list of flux points that passed various thresholds into a table of flares.

    Padding is the number of points to disregard near edges (including internal)

    Expects columns "pass_fdr" and "pass_std" in ws_stat_table.

    Expects columns "time" and "detrended_flux" in lc_table.

    Returns a table of flares with columns "start_ind", "end_ind", "start_time", "end_time", "peak_ind", and "max_flux".
    """

    in_flare_array = np.concatenate((ws_stat_table["pass_fdr"] & ws_stat_table["pass_std"], [True])) & \
                     np.concatenate(([True], ws_stat_table["pass_fdr"] & ws_stat_table["pass_std"]))
    lc_table["flaring"] = in_flare_array

    all_flares = []
    part_start_ind = 0
    for partition in np.unique(lc_table["partition"]):
        part_lc = lc_table[lc_table["partition"]==partition]
        
        in_flare = False
        start_ind = -1
        end_ind = -1
        flares = []
        for i,row in enumerate(part_lc):
    
            if i <= end_ind: # A calculated flare has extending to this indice
                continue
    
            if row["flaring"]: # flux was marked as flaring
                if not in_flare: # Get the flare started
                    start_ind = i
                    in_flare = True
            else: # flux was not marked as flaring
                if in_flare: # End the flare, find edges, catalog
                    in_flare = False                
                    start_ind, end_ind = find_flare_edges(part_lc["flux_detrended"], start_ind, i)

                    if padding > 0:
                        # Check if flare is within padding points of an edge
                        if (start_ind < padding) or ((len(part_lc) - end_ind) < padding):
                            continue  # Don't log flare, too near edge

                    # Check if we need to just add this to the previous flare
                    if len(flares) and (start_ind) == flares[-1]["end_ind"]:
                        start_ind = flares[-1]["start_ind"] # set the ends to the full flare
                        flares = flares[:-1] # remove the previous flare
                
                    max_flux = np.max(part_lc["flux_detrended"][start_ind:end_ind])
                    peak_ind = np.where(part_lc["flux_detrended"] == max_flux)[0][0]
            
                    flares.append({"start_ind":start_ind, "end_ind":end_ind,
                                   "start_time":part_lc["time"][start_ind],
                                   "end_time":part_lc["time"][end_ind], 
                                   "peak_ind":peak_ind, "max_flux":max_flux})
        # end of partition
        for ent in flares:
            ent["start_ind"] += part_start_ind
            ent["end_ind"] += part_start_ind
            ent["peak_ind"] += part_start_ind

        all_flares += flares
        part_start_ind += len(part_lc)
            
    # remove the column we added
    lc_table.remove_column("flaring")
    
    return Table(all_flares)
    
