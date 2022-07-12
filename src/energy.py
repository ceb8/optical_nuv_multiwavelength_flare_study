import numpy as np

from astropy import units as u
from astropy.time import Time

from scipy.optimize import curve_fit
from scipy.stats import zscore

import arhap

R_KEP, R_ERR = 2.32, 0.03 # Yuan et al

def line_func(x, m, b):
    """ Simple slope-intercept line function"""
    return m*x + b


def calc_qui_flux_parms(lc_table, flare_inds, padding=5, nump_pts=10):
    """
    Given a light curve and flare edge indexes, calculate the parameters
    for the quiescent flux line.
    """

    
    qui_times = np.concatenate((lc_table['time'].jd[flare_inds[0]-(nump_pts+padding):flare_inds[0]-padding],
                                lc_table['time'].jd[flare_inds[1]+padding:flare_inds[1]+(nump_pts+padding)]))

    qui_flux = np.concatenate((lc_table['flux'][flare_inds[0]-(nump_pts+padding):flare_inds[0]-padding],
                               lc_table['flux'][flare_inds[1]+padding:flare_inds[1]+(nump_pts+padding)]))

    qui_flux_err = np.concatenate((lc_table['flux_err'][flare_inds[0]-(nump_pts+padding):flare_inds[0]-padding],
                                   lc_table['flux_err'][flare_inds[1]+padding:flare_inds[1]+(nump_pts+padding)]))    

    # This shouldn't be necessary but I was getting a weird error about masked columns
    qui_times = np.array(qui_times)
    qui_flux = np.array(qui_flux)
    qui_flux_err = np.array(qui_flux_err)
    
    # Removing outliers (any pts more than 2.75 stds above/below mean)
    outliers = (np.abs(zscore(qui_flux)) > 2.5)
    qui_times = qui_times[~outliers]
    qui_flux = qui_flux[~outliers]
    qui_flux_err = qui_flux_err[~outliers]

    popt, pcov = curve_fit(line_func, qui_times, qui_flux, sigma=qui_flux_err)
    m, b = popt
    
    N = len(qui_times)
    std = np.sqrt((1/(N-2))*sum((qui_flux - b - m*qui_times)**2)) # std error from linearity
    
    return (m, b, std)


def kepler_count_to_flux(cnt):
    """
    Take Kepler flux (electron/sec) and translate it into actual flux (erg/s/cm^2)
    """
    mag = arhap.kepler_count_to_mag(cnt)
    return arhap.abmag_to_flux(mag, arhap.Kepler_lamda, arhap.Kepler_fwhm)


def get_fluence(lc_table, flare, star_prop):
    """
    Given a light curve and flare properties return the flare's fluence and associated error (erg/cm^2).
    """
    flare_points = lc_table[flare["start_ind"]:flare["end_ind"]+1]
    
    flux = kepler_count_to_flux(flare_points["flux"])
    flux_err_hi = kepler_count_to_flux(flare_points["flux"] + flare_points["flux_err"]) - flux
    flux_err_lo = flux - kepler_count_to_flux(flare_points["flux"] - flare_points["flux_err"])

    qui_flux_kep_units = line_func(flare_points["time"].jd, flare["qui_m"],flare["qui_b"])
    qui_flux = kepler_count_to_flux(qui_flux_kep_units)
    qui_flux_err_hi = kepler_count_to_flux(qui_flux_kep_units + flare["qui_std"]) - qui_flux
    qui_flux_err_lo = qui_flux - kepler_count_to_flux(qui_flux_kep_units - flare["qui_std"])
    
    fluence = sum(flux - qui_flux) * arhap.Kepler_lc_exptime
    fluence_err = np.sqrt(sum((flux_err_hi)**2) + sum((flux_err_lo)**2) +
                          sum((qui_flux_err_hi)**2) + sum((qui_flux_err_lo)**2)) * arhap.Kepler_lc_exptime

    # Adding extinction correction
    ext_corr = 10**(R_KEP * star_prop["E(B-V)"] * 2 / 5)
    ext_err = (R_KEP * star_prop["E(B-V)"] / 25) * ext_corr * R_ERR
    
    fluence *= ext_corr
    fluence_err = fluence * np.sqrt((fluence_err/(fluence/ext_corr))**2 + (ext_err/ext_corr)**2)
    
    return fluence, fluence_err


def get_kepler_energy(fluence, fluence_err, star_prop):
    """
    Given a fluence, fluence error, and appropriate stellar properties calculated the energy in the 
    Kepler waveband (not bolometric).
    """
    dist = star_prop["r_est"]*u.pc
    d_err_hi = star_prop['r_hi']*u.pc - dist
    d_err_lo = dist - star_prop['r_lo']*u.pc
    
    E =  (4*np.pi*fluence*dist**2).to(u.erg)  
    E_err_hi = E*np.sqrt((2*d_err_hi/dist)**2 + (fluence_err/fluence)**2) 
    E_err_lo = E*np.sqrt((2*d_err_lo/dist)**2 + (fluence_err/fluence)**2)
    
    return E, E_err_hi, E_err_lo


def percent_recovery_above_E(recovered_flares, injected_flares, Emin):
    """
    Returns the percent of flares recovered above the given Emin.
    """
    
    if sum(injected_flares["flare_eng"] >= Emin) == 0:
        return 1.  # if no flares to recover, recovered all of them
    return sum(recovered_flares["inj_flare_eng"] >= Emin)/sum(injected_flares["flare_eng"] >= Emin) 



def percent_recovery_above_P(recovered_flares, injected_flares, Pmin):
    """
    Returns the percent of flares recovered above the given Pmin.
    """
    
    if sum(injected_flares["f_peak_kep"] >= Pmin) == 0:
        return 1.  # if no flares to recover, recovered all of them
    return sum(recovered_flares["peak_kep_flux"] >= Pmin)/sum(injected_flares["f_peak_kep"] >= Pmin) 



def max_kepler_energy(deltf_max_kep, f_qui_kep, dist, e_bv=0.0):
    """
    Takes a max delta flux and quiescent flux in electrons/sec and calculates the
    Kepler band flare energy that results with in that delta flux assuming the entirety
    of the flare is contained within a single kepler long cadence exposure.

    Arguments must be given with units.
    """

    deltf_max = kepler_count_to_flux(f_qui_kep+deltf_max_kep) - kepler_count_to_flux(f_qui_kep)
    
    fluence = deltf_max*arhap.Kepler_lc_exptime
    ext_corr = 10**(R_KEP * e_bv * 2 / 5)
    fluence *= ext_corr
    
    if isinstance(dist, float):
        dist *= u.pc
    E =  (4*np.pi*fluence*dist**2).to(u.erg)  
    
    return E
