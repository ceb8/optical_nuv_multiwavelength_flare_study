
import numpy as np

from astropy import units as u
from astropy.time import Time

from random import uniform

from scipy.optimize import newton

import arhap


def inverse_cdf(x):
    """
    Inverse cumulative distribution function, for the flare
    distribution from Brasseur et al. 2019.
    For use in inverse transform sampling.
    """

    Emin = 3*10**31
    alpha = 1.72

    return Emin*x**(1/(1-alpha))


def make_flare_function(f_peak, t_12, t_peak):
    """
    Makes a synthetic flare using the Davenport model 
    (https://ui.adsabs.harvard.edu/abs/2014ApJ...797..122D/abstract).

    Note: This breaks if f_peak is a quantity, idk why.

    Parameters
    -----------
    f_peak : float
        Peak flux of the flare
    t_12 : float, u.Quantity
        The characteristic timescale (FWHM) of the flare.
    t_peak : Time
        The time of the flare peak.

    Returns
    -------
    response : func
        A function that takes a time and returns the corresponding flare flux value.
    """
    
    if isinstance(f_peak, int):
        f_peak = float(f_peak)

    if isinstance(f_peak, u.quantity.Quantity):
        f_peak = f_peak.value
            
    def flare_func(t):
        """
        Synthetic flare model from Davenport et al (https://ui.adsabs.harvard.edu/abs/2014ApJ...797..122D/abstract).

        Given a time returns the corresponding flare flux value.
        """
        t_norm = ((t-t_peak)/t_12).to("")

        if t >= t_peak:
            f_norm = (0.6890 * np.exp(-1.6 * t_norm)) + (0.3030 * np.exp(-0.2783 * t_norm))
        elif t >= (t_peak - t_12):
            f_norm = 1 + (1.941 * t_norm) - (0.175 * t_norm**2) - (2.246 * t_norm**3) - (1.125 * t_norm**4)
        else:
            f_norm = 0

        return f_peak * f_norm
    
    return np.vectorize(flare_func)



def make_fluence_function(f_peak, t_12, t_peak):
    """
    Creates a function that returns the intergral of the Davenport et al 
    (https://ui.adsabs.harvard.edu/abs/2014ApJ...797..122D/abstract) model flare function
    between two given times.

    Parameters
    -----------
    f_peak : float, u.Quantity
        Peak flux of the flare
    t_12 : float, u.Quantity
        The characteristic timescale (FWHM) of the flare.
    t_peak : Time
        The time of the flare peak.

    Returns
    -------
    response : func
        A function that takes a two times and returns the fluence of the flare between those times.
        Note the resulting function strips
    """
    
    if isinstance(f_peak,int):
        f_peak = float(f_peak)
        
    if not isinstance(f_peak, u.quantity.Quantity):
        f_peak = f_peak*u.electron/u.second
        
    def integrate(t):
        """
        The integral of the Davenport et al (https://ui.adsabs.harvard.edu/abs/2014ApJ...797..122D/abstract) 
        model flare function with given f_peak, t_12, and t_peak at time t.
        """
        
        if t == np.inf: # allows us to get the total fluence of the flare
            return 0.
            
        t_norm = ((t-t_peak)/t_12).to("").value
        
        if t >= t_peak:
            fl_norm = (-0.6890/1.6 * np.exp(-1.6 * t_norm)) + (-0.3030/0.2783 * np.exp(-0.2783 * t_norm))
        elif t >= (t_peak - t_12):
            fl_norm = (1 * t_norm) + (1.941/2 * t_norm**2) - (0.175/3 * t_norm**3) - (2.246/4 * t_norm**4) - (1.125/5 * t_norm**5)
        else:
            fl_norm = 0
            
        return (f_peak*t_12).decompose().value * fl_norm
          
    # pre-calculating the integral at t_peak and t0
    t0 = (t_peak - t_12)
    tpeak_integral = integrate(t_peak)
    t0_integral = integrate(t0)
    
        
    def fluence_func(t_min, t_max):
        """
        Calculates the fluence of the flare between given times.
        """
        
        if t_min == -np.inf:
            t_min = t0
        
        if t_min < t0:
            if (t_max != np.inf) and (t_max < t0):
                return 0.
            elif (t_max != np.inf) and (t_max < t_peak):
                return integrate(t_max) - t0_integral
            else: # Entire rise phase is included in time span
                return (0 - t0_integral) + (integrate(t_max) - tpeak_integral)
        elif t_min < t_peak:
            if (t_max != np.inf) and (t_max < t_peak):
                return integrate(t_max) - integrate(t_min)
            else:
                return (0 - integrate(t_min)) + (integrate(t_max) - tpeak_integral)
        else: # entire time span is past t_peak
            return integrate(t_max) - integrate(t_min)
        
    return np.vectorize(fluence_func)
    

def compose_fluence_funcs(*fluence_funcs):
    """
    Takes a set of fluence functions (as from make_fluence_function) and returns
    a function that is the sum of all of them.
    """

    def all_flare_fluence_func(tmin, tmax):
        return sum([x(tmin, tmax) for x in fluence_funcs])

    return all_flare_fluence_func



def params_to_peak_kepler_flux(distance, flare_eng, t_12):
    """
    Function that take a stellar distance, flare energy, and flare FWHM and
    determines the max flux in electrons/second.
    
    If units are not supplied it is assumed that distance is in parsecs,
    t_12 is in seconds, and flare_eng is in ergs.
    """
    
    if not isinstance(distance, u.quantity.Quantity):
        distance = distance*u.pc
        
    if not isinstance(t_12, u.quantity.Quantity):
        t_12 = t_12*u.sec
        
    if not isinstance(flare_eng, u.quantity.Quantity):
        flare_eng = flare_eng*u.erg

    # const is (1  - (1.941/2) - (0.175/3) + (2.246/4) - (1.125/5) + (0.6890/1.6) + (0.3030/0.2783))
    peak_lum = (flare_eng/1.827044810755779/t_12).to("erg s-1")

    flux = peak_lum/(4*np.pi*distance**2)
    abmag = arhap.flux_to_abmag(flux, arhap.Kepler_lamda, arhap.Kepler_fwhm)

    return arhap.kepler_mag_to_count(abmag)


def inject_flare(kep_lc, integrated_flux, minmax = None):
    """
    Inject one or more flares into a Kepler long cadence light curve.

    minmax are optional ranges, to make it faster if you are only injecting one flare
    """
    exptime_12 = 812.67339194145*u.second # 1/2 Kepler long cadence exposure time in seconds

    bin_starts = kep_lc["time"] - exptime_12
    bin_ends = kep_lc["time"] + exptime_12

    if minmax:
        min_ind = arhap.find_nearest(bin_starts.jd, minmax[0].jd) - 1
        if min_ind < 0: min_ind = 0
        max_ind = arhap.find_nearest(bin_ends.jd, minmax[1].jd) + 1
        if max_ind == len(bin_ends): max_ind -= 1
        
        kep_lc["flux"][min_ind: max_ind] += integrated_flux(bin_starts[min_ind: max_ind],
                                                            bin_ends[min_ind: max_ind]) / (2*exptime_12.value)
    else:
        kep_lc["flux"] += integrated_flux(bin_starts, bin_ends) / (2*exptime_12.value)    



def calculate_flare_duration(mean_err, f_peak, t_12):
    """
    Given peak flux, and fwhm work out how long the flare lasts within error.
    """
 
    Fmin = (4*mean_err)/f_peak.value
    guess = (np.log(Fmin/0.948)/(-0.965) + np.log(Fmin/0.322)/(-0.290))*t_12 + t_12
    guess = guess.to('second').value
    
    if guess < 0:
        return np.nan*u.second  # parameters are not physically sensical for a flare
    
    def flare_func(t_decay):  # t_decay in sec
        int_func = make_flare_function(f_peak, t_12, Time(0,format='jd'))
        return int_func(Time(t_decay*u.second,format='jd'))
    
    
    return newton(flare_func, guess, rtol=2*mean_err, maxiter=3)*u.second + t_12



def make_flare_contour(t_12_list, E, dist, mean_err, fqui):
    """
    Given a list of FWHM and an energy, plus some star qualities,
    return a associated list of durations in minutes and peak fluxes in 
    electrons/sec
    """
    
    f_peak_list = params_to_peak_kepler_flux(dist, E, t_12_list)
    kep_f_peak_list = f_peak_list/fqui
        
    duration_list = []
    for peak, t_12 in zip(f_peak_list, t_12_list):
        duration_list.append(calculate_flare_duration(mean_err, peak, t_12).to(u.min).value)
        
    return np.array(duration_list), kep_f_peak_list  
