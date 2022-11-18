# -*- coding: iso-8859-1 -*-
"""
    Created on Nov 16 2021
    
    Description: utilities for spirou data reduction
    
    @authors:  Eder Martioli <emartioli@lna.br>, <martioli@iap.fr>
    
    Laboratorio Nacional de Astrofisica, Brazil
    Institut d'Astrophysique de Paris, France

    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import os,sys
import numpy as np

from copy import deepcopy
import matplotlib.pyplot as plt

import astropy.io.fits as fits

from scipy import constants

from astropy.io import ascii

import sophielib

#from astropy.time import Time
#import astropy.units as u
from scipy.interpolate import interp1d

from scipy import optimize
import scipy.interpolate as sint
import scipy.signal as sig
import ccf_lib
#import ccf2rv

def load_array_of_sophie_spectra(inputdata, rvfile="", apply_berv=True, silent=True, plot=False, verbose=False) :

    loc = {}
    loc["input"] = inputdata

    if silent :
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    if rvfile == "":
        #print("WARNING: RV file has not been provided ...")
        #print("*** Setting all source RVs equal to zero ***")
        rvbjds, rvs, rverrs = np.zeros(len(inputdata)), np.zeros(len(inputdata)), np.zeros(len(inputdata))
    else :
        loc["rvfile"] = rvfile

        if verbose :
            print("Reading RVs from file:",rvfile)
        
        rvbjds, rvs, rverrs = read_rv_time_series(rvfile, out_in_mps=False)

        if len(rvs) != len(inputdata):
            if verbose :
                print("WARNING: size of RVs is different than number of input *t.fits files")
                print("*** Ignoring input RVs and setting all source RVs equal to zero ***")
            rvbjds, rvs, rverrs = np.zeros(len(inputdata)), np.zeros(len(inputdata)), np.zeros(len(inputdata))
    #---
    loc["source_rv"] = np.nanmedian(rvs)

    spectra = []
    speed_of_light_in_kps = constants.c / 1000.

    if plot :
        bjd, snr, airmass, berv = [], [], [], []

    for i in range(len(inputdata)) :
        
        spectrum = sophielib.load_spectrum(inputdata[i])
        
        # set source RVs
        spectrum['FILENAME'] = inputdata[i]
        spectrum["source_rv"] = rvs[i]
        spectrum["rvfile"] = rvfile
        spectrum['RV'] = rvs[i]
        spectrum['RVERR'] = rverrs[i]

        hdr = spectrum["header"]
        
        spectrum['OBJECT'] = hdr['HIERARCH OHP TARG NAME']
        spectrum['DATE'] = hdr['HIERARCH OHP OBS DATE START']
        spectrum['BJD'] = hdr['HIERARCH OHP DRS BJD']
        spectrum['BERV'] = hdr['HIERARCH OHP DRS BERV']
        spectrum['AIRMASS'] = np.nan
        spectrum['EXPTIME'] = hdr['HIERARCH OHP CCD DKTM']
        spectrum['SNR'] = hdr['HIERARCH OHP DRS CAL EXT SN36']
        
        
        hdr['MJDATE'] = spectrum['BJD'] -  2400000.5
        hdr['DATE'] = hdr['HIERARCH OHP OBS DATE START']
        hdr['OBJECT'] = hdr['HIERARCH OHP TARG NAME']
        hdr['BJD'] = hdr['HIERARCH OHP DRS BJD']
        hdr['SNR'] = hdr['HIERARCH OHP DRS CAL EXT SN36']

        if plot :
            if i == 0 :
                objectname = spectrum['OBJECT']
            bjd.append(spectrum['BJD'])
            snr.append(spectrum['SNR'])
            airmass.append(spectrum['AIRMASS'])
            berv.append(spectrum['BERV'])

        if verbose :
            print("Spectrum ({0}/{1}): {2} OBJ={3} BJD={4:.6f} SNR={5:.1f} EXPTIME={6:.0f}s BERV={7:.3f} km/s".format(i+1,len(inputdata),inputdata[i],spectrum['OBJECT'],spectrum['BJD'],spectrum['SNR'],spectrum['EXPTIME'],spectrum['BERV']))

        if apply_berv :
            vel_shift = spectrum['RV'] - spectrum['BERV']
        else :
            vel_shift = spectrum['RV']

        wl = deepcopy(spectrum["wl"])
        
        wlc = 0.5 * (wl[0] + wl[-1])

        # relativistic calculation
        wl_stellar_frame = wl * np.sqrt((1-vel_shift/speed_of_light_in_kps)/(1+vel_shift/speed_of_light_in_kps))
        
        #wl_stellar_frame = wl / (1.0 + vel_shift / speed_of_light_in_kps)
        vel = speed_of_light_in_kps * ( wl_stellar_frame / wlc - 1.)

        spectrum['wl_sf'] = wl_stellar_frame
        spectrum['vels'] = vel
        
        #keep = (wl>655) & (wl<658)
        ##plt.plot(wl[keep],spectrum['flux'][keep])
        #plt.plot(wl_stellar_frame[keep],spectrum['flux'][keep])

        spectra.append(spectrum)
        
    #plt.show()
    #exit()
    
    loc["spectra"] = spectra

    if plot :
        bjd = np.array(bjd)
        snr = np.array(snr)
        airmass = np.array(airmass)
        berv = np.array(berv)
        
        fig, axs = plt.subplots(3, sharex=True)
        fig.suptitle('{} spectra of {}'.format(len(inputdata), objectname))
        axs[0].plot(bjd, snr, '-', color="orange",label="SNR")
        axs[0].set_ylabel('SNR')
        axs[0].legend()

        axs[1].plot(bjd, airmass, '--', color="olive",label="Airmass")
        axs[1].set_ylabel('Airmass')
        axs[1].legend()

        axs[2].plot(bjd, berv, ':', color="darkblue",label="BERV")
        axs[2].set_xlabel('BJD')
        axs[2].set_ylabel('BERV [km/s]')
        axs[2].legend()
        
        plt.show()
    
    return loc


def get_wlmin_wlmax(spectra) :

    speed_of_light_in_kps = constants.c / 1000.
    
    # find minimum and maximum wavelength for valid (not NaN) data
    wlmin, wlmax = -1e20, +1e20

    for i in range(spectra['nspectra']) :
        minwl_sf = np.nanmin(spectra["waves_sf"][i])
        maxwl_sf = np.nanmax(spectra["waves_sf"][i])

        if minwl_sf > wlmin :
            wlmin = minwl_sf
            
        if maxwl_sf < wlmax :
            wlmax = maxwl_sf

    spectra["wlmin"] = wlmin
    spectra["wlmax"] = wlmax

    return spectra

def read_rv_time_series(filename, out_in_mps=True) :
    """
        Description: function to read RV data from *.rdb file
        """
    rvdata = ascii.read(filename, data_start=2)
    bjd = np.array(rvdata['rjd']) + 2400000.
    if out_in_mps :
        rv, erv = 1000. * np.array(rvdata["vrad"]), 1000. * np.array(rvdata["svrad"])
    else :
        rv, erv = np.array(rvdata["vrad"]), np.array(rvdata["svrad"])

    return np.array(bjd, dtype='float'), np.array(rv, dtype='float'), np.array(erv, dtype='float')



def get_spectral_data(array_of_spectra, ref_index=0, verbose=False) :
    """
        Description: this function loads an array of spectra
    """
    if verbose :
        print("Loading data")
    
    loc = {}

    spectra = array_of_spectra["spectra"]

    filenames, dates = [], []
    bjds, airmasses, rvs, rverrs, bervs = [], [], [], [], []
    snrs = []
    
    ref_spectrum = spectra[ref_index]

    nspectra = len(spectra)
    loc['nspectra'] = nspectra

    waves, waves_sf, vels = [], [], []
    fluxes, fluxerrs, orders = [], [], []
    hdr = []

    for i in range(nspectra) :
        
        spectrum = spectra[i]

        if verbose:
            print("Loading input spectrum {0}/{1} : {2}".format(i+1,nspectra,spectrum['FILENAME']))
            
        filenames.append(spectrum['FILENAME'])
        hdr.append(spectrum['header'])
        dates.append(spectrum['DATE'])
            
        bjds.append(spectrum['BJD'])
        airmasses.append(spectrum['AIRMASS'])
        rvs.append(spectrum['RV'])
        rverrs.append(spectrum['RVERR'])
        bervs.append(spectrum['BERV'])
        snrs.append(spectrum['SNR'])
        
        waves.append(spectrum['wl'])
        waves_sf.append(spectrum['wl_sf'])
        vels.append(spectrum['vels'])
        fluxes.append(spectrum['flux'])
        fluxerrs.append(spectrum['fluxerr'])

    bjds  = np.array(bjds, dtype=float)
    airmasses  = np.array(airmasses, dtype=float)
    rvs  = np.array(rvs, dtype=float)
    rverrs  = np.array(rverrs, dtype=float)
    bervs  = np.array(bervs, dtype=float)
    snrs  = np.array(snrs, dtype=float)

    loc["header"] = hdr
    loc["filenames"] = filenames

    loc["bjds"] = bjds
    loc["airmasses"] = airmasses
    loc["rvs"] = rvs
    loc["rverrs"] = rverrs
    loc["bervs"] = bervs
    loc["snrs"] = snrs
    
    loc["waves"] = waves
    loc["waves_sf"] = waves_sf
    loc["vels"] = vels
    loc["fluxes"] = fluxes
    loc["fluxerrs"] = fluxerrs

    loc = get_wlmin_wlmax(loc)
    
    return loc


def get_gapfree_windows(spectra, max_vel_distance=3.0, min_window_size=120., fluxkey="fluxes", velkey="vels", wavekey="waves", verbose=False) :
    
    windows = []
    
    if verbose :
        print("Calculating windows with size > {0:.0f} km/s and with gaps < {1:.1f} km/s".format(min_window_size,max_vel_distance))

    for i in range(spectra['nspectra']) :

        nanmask = np.isfinite(spectra[fluxkey][i])
        vels = spectra[velkey][i]
        wl = spectra[wavekey][i]

        if len(vels[nanmask]) > min_window_size / 4.0 :

            dv = np.abs(vels[nanmask][1:] - vels[nanmask][:-1])
            
            gaps = dv > max_vel_distance
        
            window_v_ends = np.append(vels[nanmask][:-1][gaps],vels[nanmask][-1])
            window_v_starts = np.append(vels[nanmask][0],vels[nanmask][1:][gaps])

            window_size = np.abs(window_v_ends - window_v_starts)
            good_windows = window_size > min_window_size

            window_wl_ends = np.append(wl[nanmask][:-1][gaps],wl[nanmask][-1])
            window_wl_starts = np.append(wl[nanmask][0],wl[nanmask][1:][gaps])

            loc_windows = np.array((window_wl_starts[good_windows],window_wl_ends[good_windows])).T
        else :
            loc_windows = np.array([])

        windows.append(loc_windows)

    # save window function
    spectra["windows"] = windows

    return spectra


def set_common_wl_grid(spectra, vel_sampling=0.5, verbose=False) :

    if verbose :
        print("Setting a common wavelength grid for all spectra in the time series ... ")
    if "wlmin" not in spectra.keys() or "wlmax" not in spectra.keys():
        print("ERROR: function set_common_wl_grid() requires keywords wlmin and wlmax in input spectra, exiting.. ")
        exit()

    speed_of_light_in_kps = constants.c / 1000.
    drv = 1.0 + vel_sampling / speed_of_light_in_kps
    drv_neg = 1.0 - vel_sampling / speed_of_light_in_kps
    
    wlmin = spectra["wlmin"]
    wlmax = spectra["wlmax"]
        
    common_wl = np.array([])
    wl = wlmin
    while wl < wlmax * drv_neg :
        wl *= drv
        common_wl = np.append(common_wl, wl)
        
    wlc = (common_wl[0]+common_wl[-1])/2

    common_vel = speed_of_light_in_kps * ( common_wl / wlc - 1.)
        
    spectra["common_vel"] = common_vel
    spectra["common_wl"] = common_wl

    return spectra


# function to interpolate spectrum
def interp_spectrum(wl_out, wl_in, flux_in, good_windows, kind='cubic') :

    flux_out = np.full_like(wl_out, np.nan)

    for w in good_windows :

        mask = wl_in >= w[0]
        mask &= wl_in <= w[1]

        wl_in_copy = deepcopy(wl_in)
        flux_in_copy = deepcopy(flux_in)

        # create interpolation function for input data
        f = interp1d(wl_in_copy[mask], flux_in_copy[mask], kind=kind)

        wl1, wl2 = w[0], w[1]

        if wl1 < wl_in[mask][0] :
            wl1 = wl_in[mask][0]
        if wl2 > wl_in[mask][-1] :
            wl2 = wl_in[mask][-1]

        out_mask = wl_out > wl1
        out_mask &= wl_out < wl2

        # interpolate data
        flux_out[out_mask] = f(wl_out[out_mask])

    return flux_out


def resample_and_align_spectra(spectra, interp_kind='cubic', plot=False, verbose=False) :

    if "common_wl" not in spectra.keys() :
        print("ERROR: function resample_and_align_spectra() requires keyword common_wl in input spectra, exiting.. ")
        exit()
    
    aligned_waves = []
    
    sf_fluxes, sf_fluxerrs = [], []
    rest_fluxes, rest_fluxerrs = [], []

    common_wl = spectra['common_wl']
        
    for i in range(spectra['nspectra']) :
        if verbose :
            print("Aligning spectrum {} of {}".format(i+1,spectra['nspectra']))
        if "windows" in spectra.keys() :
            windows = spectra["windows"][i]
        else :
            windows = [[common_wl[0],common_wl[-1]]]
        keep = np.isfinite(spectra["fluxes"][i])

        flux = spectra["fluxes"][i][keep]
        fluxerr = spectra["fluxerrs"][i][keep]
            
        wl_sf = spectra["waves_sf"][i][keep]
        wl_rest = spectra["waves"][i][keep]

        sf_flux = interp_spectrum(common_wl, wl_sf, flux, windows, kind=interp_kind)
        #sf_fluxerr = interp_spectrum(common_wl, wl_sf, fluxerr, windows, kind=interp_kind)
        rest_flux = interp_spectrum(common_wl, wl_rest, flux, windows, kind=interp_kind)
        #rest_fluxerr = interp_spectrum(common_wl, wl_rest, fluxerr, windows, kind=interp_kind)
        sf_fluxerr = rest_fluxerr = common_wl * 0.
        
        aligned_waves.append(common_wl)

        sf_fluxes.append(sf_flux)
        sf_fluxerrs.append(sf_fluxerr)
        rest_fluxes.append(rest_flux)
        rest_fluxerrs.append(rest_fluxerr)

        if plot :
            #plt.errorbar(wl_rest, flux, yerr=fluxerr, fmt=".", lw=0.3, alpha=0.6)
            plt.errorbar(wl_sf, flux, yerr=fluxerr, fmt=".", lw=0.3, alpha=0.6)

            for w in windows:
                plt.vlines(w, [np.min(flux),np.min(flux)], [np.max(flux),np.max(flux)], color = "r", ls="--")
    if plot :
        plt.show()

    spectra["aligned_waves"] = aligned_waves

    spectra["sf_fluxes"] = sf_fluxes
    spectra["sf_fluxerrs"] = sf_fluxerrs
    spectra["rest_fluxes"] = rest_fluxes
    spectra["rest_fluxerrs"] = rest_fluxerrs

    return spectra


def reduce_spectra(spectra, nsig_clip=0.0, combine_by_median=False, combine_final_by_median=False, subtract=True, fluxkey="fluxes", fluxerrkey="fluxerrs", wavekey="wl", update_spectra=False, plot=False, verbose=False) :
    
    signals, ref_snrs, noises,  orders = [], [], [], []
    rel_noises = []
    snrs, snrs_err = [], []
    template = []

    if subtract :
        sub_flux_base = 0.0
    else :
        sub_flux_base = 1.0
    
    if verbose:
        print("Reducing spectra ...")

    # get mean signal before applying flux corrections
    median_signals = []
    for i in range(spectra['nspectra']) :
        median_signals.append(np.nanmedian(spectra[fluxkey][i]))
    median_signals = np.array(median_signals)

    # 1st pass - to build template for each order and subtract out all spectra by template
    template = calculate_template(spectra[fluxkey], wl=spectra[wavekey], fit=True, median=combine_by_median, subtract=True, sub_flux_base=sub_flux_base, verbose=False, plot=False)

    # Recover fluxes already shifted and re-scaled to match the template
    fluxes = template["flux_arr_sub"] + template["flux"] - sub_flux_base

    # 2nd pass - to build template from calibrated fluxes
    template = calculate_template(fluxes, wl=spectra[wavekey], fit=True, median=combine_by_median, subtract=subtract, sub_flux_base=sub_flux_base, verbose=False, plot=False)

    # apply sigma-clip using template and median dispersion in time as clipping criteria
    # bad values can either be replaced by the template values, by interpolated values or by NaNs
    if nsig_clip > 0 :
        template = sigma_clip(template, nsig=nsig_clip, interpolate=False, replace_by_model=False, sub_flux_base=sub_flux_base, plot=False)
        #template = sigma_clip_remove_bad_columns(template, nsig=nsig_clip, plot=False)

    # Recover fluxes already shifted and re-scaled to match the template
    if subtract :
        fluxes = template["flux_arr_sub"] + template["flux"] - sub_flux_base
    else:
        fluxes = template["flux_arr_sub"] * template["flux"]

    # 3rd pass - Calculate a final template combined by the mean
    template = calculate_template(fluxes, wl=spectra[wavekey], fit=True, median=combine_final_by_median, subtract=subtract, sub_flux_base=sub_flux_base, verbose=False, plot=plot)

    # save number of spectra in the time series
    template['nspectra'] = spectra['nspectra']
    
    if update_spectra :
        for i in range(spectra['nspectra']) :
            flux = np.full_like(template["flux_arr_sub"][i], np.nan)
        
            outkeep = (np.isfinite(template["flux_arr_sub"][i])) & (np.isfinite(template["flux"]))
            # Recover fluxes already shifted and re-scaled to match the template
            if subtract :
                flux[outkeep] = template["flux_arr_sub"][i][outkeep] + template["flux"][outkeep] - sub_flux_base
            else:
                flux[outkeep] = template["flux_arr_sub"][i][outkeep] * template["flux"][outkeep]
            
            fluxerr = np.zeros_like(template["flux_arr_sub"][i]) + template["fluxerr"]
            
            spectra[fluxkey][i] = flux
            spectra[fluxerrkey][i] = fluxerr

    return spectra, template


#################################################################################################
def calculate_template(flux_arr, wl=[], fit=False, median=True, subtract=False, sub_flux_base=1.0, min_npoints=100, verbose=False, plot=False, pfilename=""):
    """
        Compute the mean/median template spectrum along the time axis and divide/subtract
        each exposure by the mean/median
        
        Inputs:
        - flux_arr: 2D flux matrix (N_exposures, N_wavelengths)
        - wl: 1D wavelength array (N_wavelengths)
        - fit: boolean to fit median spectrum to each observation before normalizing it
        - median: boolean to calculate median instead of mean
        - subtract: boolean to subtract instead of dividing out spectra by the mean/median template

        Outputs:
        - loc: python dict containing all products
    """
    
    loc = {}

    loc["fit"] = fit
    loc["median"] = median
    loc["subtract"] = subtract
    loc["pfilename"] = pfilename

    if len(wl) == 0:
        x = np.arange(len(flux_arr[0]))
    else :
        x = wl

    if verbose :
        print("Calculating template out of {0} input spectra".format(len(flux_arr)))
    
    if median :
        # median combine
        flux_template = np.nanmedian(flux_arr,axis=0)
    else :
        # mean combine
        flux_template = np.nanmean(flux_arr,axis=0)
        #flux_template = np.average(flux_arr,axis=0, weights=weights)

    if fit :
        flux_calib = []
        flux_fit = []
        
        shift_arr = []
        scale_arr = []
        quadratic_arr = []

        def flux_model (coeffs, template, wave):
            outmodel = coeffs[2] * wave * wave + coeffs[1] * template + coeffs[0]
            return outmodel
        
        def errfunc (coeffs, fluxes, xx) :
            nanmask = ~np.isnan(fluxes)
            residuals = fluxes[nanmask] - flux_model (coeffs, flux_template[nanmask], xx[nanmask])
            return residuals

        for i in range(len(flux_arr)):
            
            nanmask = ~np.isnan(flux_arr[i])
            
            if len(flux_arr[i][nanmask]) > min_npoints :
                #guess = [0.0001, 1.001]
                guess = [0.0001, 1.001, 0.0000001]
                pfit, success = optimize.leastsq(errfunc, guess, args=(flux_arr[i], x))
            else :
                pfit = [0.,1.,0.]

            flux_template_fit = flux_model(pfit, flux_template, x)
            flux_fit.append(flux_template_fit)

            shift_arr.append(pfit[0])
            scale_arr.append(pfit[1])
            quadratic_arr.append(pfit[2])

            #flux_calib_loc = (flux_arr[i] - pfit[0]) / pfit[1]
            flux_calib_loc = (flux_arr[i] - pfit[2] * x * x - pfit[0]) / pfit[1]
            flux_calib.append(flux_calib_loc)

        loc["shift"] = np.array(shift_arr, dtype=float)
        loc["scale"] = np.array(scale_arr, dtype=float)
        loc["quadratic"] = np.array(quadratic_arr, dtype=float)

        flux_calib = np.array(flux_calib, dtype=float)
        flux_fit = np.array(flux_fit, dtype=float)

        # Compute median on all spectra along the time axis
        if median :
            flux_template_new = np.nanmedian(flux_calib,axis=0)
        else :
            flux_template_new = np.nanmean(flux_calib,axis=0)
            #flux_template_new = np.average(flux_calib,axis=0, weights=weights)

        flux_template = flux_template_new
        if subtract :
            flux_arr_sub = flux_calib - flux_template + sub_flux_base
        else :
            flux_arr_sub = flux_calib / flux_template

        residuals = flux_calib - flux_template
        flux_template_medsig = np.nanmedian(np.abs(residuals),axis=0) / 0.67449
        loc["flux_arr"] = flux_calib
    else :
        # Divide or subtract each ccf by ccf_med
        if subtract :
            flux_arr_sub = flux_arr - flux_template + sub_flux_base
        else :
            flux_arr_sub = flux_arr / flux_template

        residuals = flux_arr - flux_template
        flux_template_medsig = np.nanmedian(np.abs(residuals),axis=0) / 0.67449
        loc["flux_arr"] = flux_arr

    loc["flux"] = flux_template
    loc["fluxerr"] = flux_template_medsig
    loc["wl"] = x
    loc["flux_arr_sub"] = flux_arr_sub
    loc["flux_residuals"] = residuals
    loc["snr"] = flux_arr / flux_template_medsig

    loc["template_source"] = "data"
    
    template_nanmask = ~np.isnan(flux_template)
    template_nanmask &= ~np.isnan(flux_template_medsig)
    
    if len(flux_template_medsig[template_nanmask]) :
        loc["fluxerr_model"] = fit_continuum(x, flux_template_medsig, function='polynomial', order=5, nit=5, rej_low=2.5, rej_high=2.5, grow=1, med_filt=0, percentile_low=0., percentile_high=100.,min_points=10, xlabel="wavelength", ylabel="flux error", plot_fit=False,silent=True)
    else :
        loc["fluxerr_model"] = np.full_like(x,np.nan)

    if plot :
        plot_template_products(loc, pfilename=pfilename)
    
    return loc


def sigma_clip(template, nsig=3.0, interpolate=False, replace_by_model=True, sub_flux_base=1.0, plot=False) :
    
    out_flux_arr = np.full_like(template["flux_arr"], np.nan)
    out_flux_arr_sub = np.full_like(template["flux_arr_sub"], np.nan)

    for i in range(len(template["flux_arr"])) :
        sigclipmask = np.abs(template["flux_residuals"][i]) > (nsig * template["fluxerr_model"])
        if plot :
            plt.plot(template["wl"], template["flux_residuals"][i], alpha=0.3)
            if len(template["flux_residuals"][i][sigclipmask]) :
                plt.plot(template["wl"][sigclipmask], template["flux_residuals"][i][sigclipmask], "bo")
    
        # set good values first
        out_flux_arr[i][~sigclipmask] = template["flux_arr"][i][~sigclipmask]
        out_flux_arr_sub[i][~sigclipmask] = template["flux_arr_sub"][i][~sigclipmask]
    
        # now decide what to do with outliers
        if interpolate :
            if i > 0 and i < len(template["flux_arr"]) - 1 :
                out_flux_arr[i][sigclipmask] = (template["flux_arr"][i-1][sigclipmask] + template["flux_arr"][i+1][sigclipmask]) / 2.
                out_flux_arr_sub[i][sigclipmask] = (template["flux_arr_sub"][i-1][sigclipmask] + template["flux_arr_sub"][i+1][sigclipmask]) / 2.
            elif i == 0 :
                out_flux_arr[i][sigclipmask] = template["flux_arr"][i+1][sigclipmask]
                out_flux_arr_sub[i][sigclipmask] = template["flux_arr_sub"][i+1][sigclipmask]
            elif i == len(template["flux_arr"]) - 1 :
                out_flux_arr[i][sigclipmask] = template["flux_arr"][i-1][sigclipmask]
                out_flux_arr_sub[i][sigclipmask] = template["flux_arr_sub"][i-1][sigclipmask]
        
        if replace_by_model :
            out_flux_arr[i][sigclipmask] = template["flux"][sigclipmask]
            out_flux_arr_sub[i][sigclipmask] = sub_flux_base

        #if plot :
        #    plt.plot(template["wl"][sigclipmask],out_flux_arr[i][sigclipmask],'b.')

    if plot :
        plt.plot(template["wl"], nsig * template["fluxerr_model"], 'r--', lw=2)
        plt.plot(template["wl"], -nsig * template["fluxerr_model"], 'r--', lw=2)
        plt.show()
    
    template["flux_arr"] = out_flux_arr
    template["flux_arr_sub"] = out_flux_arr_sub

    return template


def plot_template_products(template, pfilename="") :

    wl = template["wl"]

    for i in range(len(template["flux_arr"])) :
        
        flux = template["flux_arr"][i]
        resids = template["flux_residuals"][i]

        if i == len(template["flux_arr"]) - 1 :
            plt.plot(wl, flux,"-", color='#ff7f0e', lw=0.6, alpha=0.6, label="SPIRou data")
            plt.plot(wl, resids,"-", color='#8c564b', lw=0.6, alpha=0.5, label="Residuals")
        else :
            plt.plot(wl, flux,"-", color='#ff7f0e', lw=0.6, alpha=0.6)
            plt.plot(wl, resids,"-", color='#8c564b', lw=0.6, alpha=0.5)

    plt.plot(template["wl"], template["flux"],"-", color="red", lw=2, label="Template spectrum")

    sig_clip = 3.0
    plt.plot(template["wl"], sig_clip * template["fluxerr"],"--", color="olive", lw=0.8)
    plt.plot(template["wl"], sig_clip * template["fluxerr_model"],"-", color="k", lw=0.8)

    plt.plot(template["wl"],-sig_clip * template["fluxerr"],"--", color="olive", lw=0.8, label=r"{0:.0f}$\sigma$ (MAD)".format(sig_clip))
    plt.plot(template["wl"],-sig_clip * template["fluxerr_model"],"-", color="k", lw=0.8, label="{0:.0f}$\sigma$ fit model".format(sig_clip))

    plt.legend(fontsize=20)
    plt.xlabel(r"$\lambda$ [nm]", fontsize=26)
    plt.ylabel(r"Flux", fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    if pfilename != "" :
        plt.savefig(pfilename, format='png')
    else :
        plt.show()
    plt.clf()
    plt.close()



def fit_continuum(wav, spec, function='polynomial', order=3, nit=5, rej_low=2.0,
    rej_high=2.5, grow=1, med_filt=0, percentile_low=0., percentile_high=100.,
                  min_points=10, xlabel="", ylabel="", plot_fit=True, verbose=False, silent=False):
    """
    Continuum fitting re-implemented from IRAF's 'continuum' function
    in non-interactive mode only but with additional options.

    :Parameters:
    
    wav: array(float)
        abscissa values (wavelengths, velocities, ...)

    spec: array(float)
        spectrum values

    function: str
        function to fit to the continuum among 'polynomial', 'spline3'

    order: int
        fit function order:
        'polynomial': degree (not number of parameters as in IRAF)
        'spline3': number of knots

    nit: int
        number of iteractions of non-continuum points
        see also 'min_points' parameter

    rej_low: float
        rejection threshold in unit of residul standard deviation for point
        below the continuum

    rej_high: float
        same as rej_low for point above the continuum

    grow: int
        number of neighboring points to reject

    med_filt: int
        median filter the spectrum on 'med_filt' pixels prior to fit
        improvement over IRAF function
        'med_filt' must be an odd integer

    percentile_low: float
        reject point below below 'percentile_low' percentile prior to fit
        improvement over IRAF function
        "percentile_low' must be a float between 0. and 100.

    percentile_high: float
        same as percentile_low but reject points in percentile above
        'percentile_high'
        
    min_points: int
        stop rejection iterations when the number of points to fit is less than
        'min_points'

    plot_fit: bool
        if true display two plots:
            1. spectrum, fit function, rejected points
            2. residual, rejected points

    verbose: bool
        if true fit information is printed on STDOUT:
            * number of fit points
            * RMS residual
    """
    if silent :
        import warnings
        warnings.simplefilter('ignore', np.RankWarning)
    
    mspec = np.ma.masked_array(spec, mask=np.zeros_like(spec))
    # mask 1st and last point: avoid error when no point is masked
    # [not in IRAF]
    mspec.mask[0] = True
    mspec.mask[-1] = True
    
    mspec = np.ma.masked_where(np.isnan(spec), mspec)
    
    # apply median filtering prior to fit
    # [opt] [not in IRAF]
    if int(med_filt):
        fspec = sig.medfilt(spec, kernel_size=med_filt)
    else:
        fspec = spec
    # consider only a fraction of the points within percentile range
    # [opt] [not in IRAF]
    mspec = np.ma.masked_where(fspec < np.percentile(fspec, percentile_low),
        mspec)
    mspec = np.ma.masked_where(fspec > np.percentile(fspec, percentile_high),
        mspec)
    # perform 1st fit
    if function == 'polynomial':
        coeff = np.polyfit(wav[~mspec.mask], spec[~mspec.mask], order)
        cont = np.poly1d(coeff)(wav)
    elif function == 'spline3':
        knots = wav[0] + np.arange(order+1)[1:]*((wav[-1]-wav[0])/(order+1))
        spl = sint.splrep(wav[~mspec.mask], spec[~mspec.mask], k=3, t=knots)
        cont = sint.splev(wav, spl)
    else:
        raise(AttributeError)
    # iteration loop: reject outliers and fit again
    if nit > 0:
        for it in range(nit):
            res = fspec-cont
            sigm = np.std(res[~mspec.mask])
            # mask outliers
            mspec1 = np.ma.masked_where(res < -rej_low*sigm, mspec)
            mspec1 = np.ma.masked_where(res > rej_high*sigm, mspec1)
            # exlude neighbors cf IRAF's continuum parameter 'grow'
            if grow > 0:
                for sl in np.ma.clump_masked(mspec1):
                    for ii in range(sl.start-grow, sl.start):
                        if ii >= 0:
                            mspec1.mask[ii] = True
                    for ii in range(sl.stop+1, sl.stop+grow+1):
                        if ii < len(mspec1):
                            mspec1.mask[ii] = True
            # stop rejection process when min_points is reached
            # [opt] [not in IRAF]
            if np.ma.count(mspec1) < min_points:
                if verbose:
                    print("  min_points %d reached" % min_points)
                break
            mspec = mspec1
            if function == 'polynomial':
                coeff = np.polyfit(wav[~mspec.mask], spec[~mspec.mask], order)
                cont = np.poly1d(coeff)(wav)
            elif function == 'spline3':
                knots = wav[0] + np.arange(order+1)[1:]*((wav[-1]-wav[0])/(order+1))
                spl = sint.splrep(wav[~mspec.mask], spec[~mspec.mask], k=3, t=knots)
                cont = sint.splev(wav, spl)
            else:
                raise(AttributeError)
    # compute residual and rms
    res = fspec-cont
    sigm = np.std(res[~mspec.mask])
    if verbose:
        print("  nfit=%d/%d" %  (np.ma.count(mspec), len(mspec)))
        print("  fit rms=%.3e" %  sigm)
    # compute residual and rms between original spectrum and model
    # different from above when median filtering is applied
    ores = spec-cont
    osigm = np.std(ores[~mspec.mask])
    if int(med_filt) and verbose:
        print("  unfiltered rms=%.3e" %  osigm)
    # plot fit results
    if plot_fit:
        # overplot spectrum and model + mark rejected points
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)
        ax1.plot(wav[~mspec.mask], spec[~mspec.mask],
            c='tab:blue', lw=1.0)
        # overplot median filtered spectrum
        if int(med_filt):
            ax1.plot(wav[~mspec.mask], fspec[~mspec.mask],
                c='tab:cyan', lw=1.0)
        ax1.scatter(wav[mspec.mask], spec[mspec.mask], s=20., marker='d',
        edgecolors='tab:gray', facecolors='none', lw=0.5)
        ax1.plot(wav, cont, ls='--', c='tab:orange')
        if nit > 0:
            # plot residuals and rejection thresholds
            fig2 = plt.figure(2)
            ax2 = fig2.add_subplot(111)
            ax2.axhline(0., ls='--', c='tab:orange', lw=1.)
            ax2.axhline(-rej_low*sigm, ls=':')
            ax2.axhline(rej_high*sigm, ls=':')
            ax2.scatter(wav[mspec.mask], res[mspec.mask],
                s=20., marker='d', edgecolors='tab:gray', facecolors='none',
                lw=0.5)
            ax2.scatter(wav[~mspec.mask], ores[~mspec.mask],
                marker='o', s=10., edgecolors='tab:blue', facecolors='none',
                lw=.5)
            # overplot median filtered spectrum
            if int(med_filt):
                ax2.scatter(wav[~mspec.mask], res[~mspec.mask],
                    marker='s', s=5., edgecolors='tab:cyan', facecolors='none',
                    lw=.2)
        if xlabel != "" :
            plt.xlabel(xlabel)
        if ylabel != "" :
            plt.ylabel(ylabel)
        plt.show()
    return cont


def normalize_spectra(spectra, template, fluxkey="fluxes", fluxerrkey="fluxerrs", cont_function='polynomial', polyn_order=4, med_filt=1, plot=False) :
    
    continuum_fluxes = []

    wl = template["wl"]
    flux = template["flux"]
    fluxerr = template["fluxerr"]
        
    keep = np.isfinite(flux)
    keep &= np.isfinite(wl)

    continuum = np.full_like(wl, np.nan)

    if len(flux[keep]) > 10 :
        continuum[keep] = fit_continuum(wl[keep], flux[keep], function=cont_function, order=polyn_order, nit=10, rej_low=1.5, rej_high=3.5, grow=1, med_filt=med_filt, percentile_low=0., percentile_high=100.,min_points=100, xlabel="wavelength", ylabel="flux", plot_fit=False, silent=True)
        
    if plot :
        plt.errorbar(wl, flux, yerr=fluxerr, fmt='.', lw=0.3, alpha=0.3, zorder=1, label="spectral data")
        #plt.scatter(wl, flux, marker='o', s=10., edgecolors='tab:blue', facecolors='none', lw=.5)
        plt.plot(wl, continuum, '-', lw=2, zorder=2, label="continuum")
    
    for i in range(spectra['nspectra']) :
        spectra[fluxkey][i] /= continuum
        spectra[fluxerrkey][i] /= continuum
        
    template["continuum"] = continuum
           
    template["flux"] /= continuum
    template["fluxerr"] /= continuum
    template["fluxerr_model"] /= continuum
    
    for j in range(len(template["flux_arr"])) :
        template["flux_arr"][j] /= continuum
        template["flux_residuals"][j] /= continuum

    if plot :
        plt.legend(fontsize=18)
        plt.xlabel(r"$\lambda$ [nm]", fontsize=20)
        plt.ylabel(r"Flux", fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.show()

    spectra["continuum_{}".format(fluxkey)] = continuum

    return spectra, template


def calculate_weights(spectra, template, normalize_weights=True, use_err_model=True, plot=False) :

    wl = template["wl"]
    flux = template["flux"]

    if use_err_model :
        fluxerr = template["fluxerr_model"]
    else :
        fluxerr = template["fluxerr"]

    nanmask = np.isfinite(flux)
    nanmask &= np.isfinite(fluxerr)
    nanmask &= flux > 0
            
    weights = np.full_like(fluxerr, np.nan)
    weights[nanmask] = 1. / (fluxerr[nanmask] * fluxerr[nanmask])

    if normalize_weights :
        normfactor = np.nanmedian(weights[nanmask])
        weights /= normfactor

    if plot :
        plt.ylim(-0.5,3.0)
        plt.scatter(wl, weights, marker='o', s=10., edgecolors='tab:red', facecolors='none', lw=.5)
        plt.plot(wl, flux, '-')
        plt.show()

    spectra["weights"] = np.array(weights, dtype=float)

    return spectra


def plot_template_products_with_CCF_mask(template, ccfmask, source_rv=0, pfilename="") :

    wl = template["wl"]

    for i in range(len(template["flux_arr"])) :
        
        flux = template["flux_arr"][i]
        resids = template["flux_residuals"][i]

        if i == len(template["flux_arr"]) - 1 :
            plt.plot(wl, flux,"-", color='#ff7f0e', lw=0.6, alpha=0.6, label="SOPHIE data", zorder=1)
            plt.plot(wl, resids,".", color='#8c564b', lw=0.2, alpha=0.2, label="Residuals", zorder=1)
        else :
            plt.plot(wl, flux,"-", color='#ff7f0e', lw=0.6, alpha=0.6, zorder=1)
            plt.plot(wl, resids,".", color='#8c564b', lw=0.2, alpha=0.2, zorder=1)

    # Plot CCF mask
    lines_wlc = ccfmask["centers"]
    lines_wei = ccfmask["weights"]
    speed_of_light_in_kps = constants.c / 1000.
    wlc_starframe = lines_wlc * (1.0 + source_rv / speed_of_light_in_kps)
    median_flux = np.nanmedian(template["flux"])
    plt.vlines(wlc_starframe, median_flux - lines_wei / np.nanmax(lines_wei), median_flux,ls="--", lw=0.7, label="CCF lines", zorder=2)
    #---------------
    
    plt.plot(template["wl"], template["flux"],"-", color="red", lw=2, label="Template spectrum", zorder=1.5)

    sig_clip = 3.0
    plt.plot(template["wl"], sig_clip * template["fluxerr"],"-", color="darkgreen", lw=2, zorder=1.1)
    #plt.plot(template["wl"], sig_clip * template["fluxerr_model"],"-", color="k", lw=0.8, zorder=2)

    plt.plot(template["wl"],-sig_clip * template["fluxerr"],"-", color="darkgreen", lw=2, label=r"{0:.0f}$\sigma$".format(sig_clip), zorder=1.1)
    #plt.plot(template["wl"],-sig_clip * template["fluxerr_model"],"-", color="k", lw=0.8, label="{0:.0f}$\sigma$ fit model".format(sig_clip), zorder=2)

    plt.legend(fontsize=16)
    plt.xlabel(r"$\lambda$ [nm]", fontsize=26)
    plt.ylabel(r"Relative flux", fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    #plt.xlim(1573,1581)
    if pfilename != "" :
        plt.savefig(pfilename, format='png')
    else :
        plt.show()
    plt.clf()
    plt.close()


    
def get_zero_drift_containers(inputdata) :
    
    drifts, drifts_err = np.zeros(len(inputdata)), np.zeros(len(inputdata))
    output = []
    for i in range(len(inputdata)) :
        hdr = fits.getheader(inputdata[i])
        loc = {}
        loc["FILENAME"] = inputdata[i] # Wavelength sol absolute CCF FP Drift [km/s]
        loc["WFPDRIFT"] = 'None' # Wavelength sol absolute CCF FP Drift [km/s]
        loc["RV_WAVFP"] = 'None' # RV measured from wave sol FP CCF [km/s]
        loc["RV_SIMFP"] = 'None' # RV measured from simultaneous FP CCF [km/s]
        loc["RV_DRIFT"] = drifts[i] # RV drift between wave sol and sim. FP CCF [km/s]
        loc["RV_DRIFTERR"] = drifts_err[i] # RV drift error between wave sol and sim. FP CCF [km/s]
        output.append(loc)
    
    return output




def reduce_timeseries_of_spectra(inputdata, ccf_mask="", object_name="", stellar_spectrum_file="", source_rv=0., max_gap_size=1.0, min_window_size=150., align_spectra=True, vel_sampling=0.5, nsig_clip = 3.0, ccf_width=150, fwhm=8.0, output_template="", fluxkey="rest_fluxes", fluxerrkey="rest_fluxerrs", apply_berv=True, normalize=False, cont_function='spline3', polyn_order=40, save_ccf=False, save_spectra=False, normalize_ccfs=True, plot_ccf=False, plot_details=False, plot=False, verbose=False) :

    # First load spectra into a container
    array_of_spectra = load_array_of_sophie_spectra(inputdata, apply_berv=apply_berv, plot=False, verbose=verbose)

    # Then load data into vector
    spectra = get_spectral_data(array_of_spectra, verbose=verbose)

    # Use wide values to avoid too much clipping at this point. This will improve the noise model
    #spectra = get_gapfree_windows(spectra, max_vel_distance=max_gap_size, min_window_size=min_window_size, fluxkey="fluxes", wavekey="waves_sf", verbose=verbose)

    # Set a common wavelength grid for all input spectra
    spectra = set_common_wl_grid(spectra, vel_sampling=vel_sampling, verbose=verbose)

    # Interpolate all spectra to a common wavelength grid
    spectra = resample_and_align_spectra(spectra, verbose=verbose, plot=plot_details)
    #spectra["aligned_waves"]
    #spectra["sf_fluxes"],spectra["sf_fluxerrs"]
    #spectra["rest_fluxes"], spectra["rest_fluxerrs"]

    spectra, template = reduce_spectra(spectra, nsig_clip=5.0, combine_by_median=True, subtract=True, fluxkey=fluxkey, fluxerrkey=fluxerrkey, wavekey="common_wl", update_spectra=True, plot=plot_details, verbose=True)

    if normalize :
        # detect continuum in the template spectrum and normalize all spectra to the same continuum
        spectra, template = normalize_spectra(spectra, template, fluxkey=fluxkey, fluxerrkey=fluxerrkey, cont_function=cont_function, polyn_order=polyn_order, med_filt=1, plot=plot)
        
    else :
        spectra, template = normalize_spectra(spectra, template, fluxkey=fluxkey, fluxerrkey=fluxerrkey, cont_function='polynomial', polyn_order=0, med_filt=1, plot=plot_details)

    # Plot template spectrum after normalization
    if plot :
        plt.plot(template["wl"], template["flux"],'-', label="Template spectrum")
        plt.legend(fontsize=18)
        plt.xlabel(r"$\lambda$ [nm]", fontsize=20)
        plt.ylabel(r"Flux", fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.show()
    
    # Calculate statistical weights based on the time series dispersion 1/sig^2
    spectra = calculate_weights(spectra, template, use_err_model=False, plot=plot_details)

    # Initialize drift containers with zeros
    drifts = get_zero_drift_containers(inputdata)
    
    # Start dealing with CCF related parameters and construction of a weighted mask
    # load science CCF parameters
    ccf_params = ccf_lib.set_ccf_params(ccf_mask)

    # update ccf width with input value
    ccf_params["CCF_WIDTH"] = float(ccf_width)

    ccfmask = ccf_lib.apply_weights_to_ccf_mask(ccf_params, template["wl"], template["flux"], template["fluxerr"], spectra["weights"], median=True, remove_lines_with_nans=True, source_rv=source_rv, verbose=True, plot=plot_details)
    
    base_header = deepcopy(array_of_spectra["spectra"][0]["header"])

    # Run ccf on the template to obtain the radial velocity of the star
    template_ccf = ccf_lib.run_ccf_eder(ccf_params, template["wl"], template["flux"], base_header, ccfmask, targetrv=source_rv, normalize_ccfs=True, output=False, plot=False, verbose=False)

    source_rv = template_ccf["header"]['RV_OBJ']
    fwhm = template_ccf["header"]['CCFMFWHM']
        
    # Apply weights to stellar CCF mask
    ccfmask = ccf_lib.apply_weights_to_ccf_mask(ccf_params, template["wl"], template["flux"], template["fluxerr"], spectra["weights"], median=True, remove_lines_with_nans=True, source_rv=source_rv, verbose=verbose, plot=plot_details)

    # plot template
    #if options.plot :
    #    plot_template_products_with_CCF_mask(template, ccfmask, source_rv=ccf_params["SOURCE_RV"], pfilename="")

    ccf_params["CCF_WIDTH"] = 8 * fwhm
    ccf_params["SOURCE_RV"] = source_rv

    if verbose :
        print("Source RV={:.4f} km/s  CCF FWHM={:.2f} km/s CCF window size={:.2f} km/s".format(source_rv,fwhm,ccf_params["CCF_WIDTH"]))

    if output_template != "" :
        if verbose :
            print("Saving template spectrum to file: {0} ".format(output_template))
        sophielib.write_spectrum_to_fits(template["wl"], template["flux"], template["fluxerr"], output_template, header=template_ccf["header"])

    ###### START CCF  #######
    if plot_ccf :
        templ_legend = "Template of {}".format(template_ccf["header"]["OBJECT"].replace(" ",""))
        plt.plot(template_ccf['RV_CCF'], template_ccf['MEAN_CCF'], "-", color='green', lw=2, label=templ_legend, zorder=2)
    
    calib_rv, drift_rv  = [], []
    mean_fwhm = []
    sci_ccf_file_list = []

    for i in range(spectra['nspectra']) :
    
        if verbose :
            print("Running CCF on file {0}/{1} -> {2}".format(i,spectra['nspectra']-1,os.path.basename(spectra['filenames'][i])))

        rv_drifts = drifts[i]

        fluxes, fluxerrs  = spectra[fluxkey][i], spectra[fluxerrkey][i]
        waves_sf = spectra["aligned_waves"][i]
        # run main routine to process ccf on science fiber
        header = array_of_spectra["spectra"][i]["header"]

        # run an adpated version of the ccf codes using reduced spectra as input
        sci_ccf = ccf_lib.run_ccf_eder(ccf_params, waves_sf, fluxes, header, ccfmask, rv_drifts=rv_drifts, filename=spectra['filenames'][i], targetrv=ccf_params["SOURCE_RV"], normalize_ccfs=normalize_ccfs, output=save_ccf, plot=False, verbose=False)

        if save_ccf :
            sci_ccf_file_list.append(os.path.abspath(sci_ccf["file_path"]))

        calib_rv.append(sci_ccf["header"]['RV_OBJ'])
        mean_fwhm.append(sci_ccf["header"]['CCFMFWHM'])
        drift_rv.append(sci_ccf["header"]['RV_DRIFT'])

        if verbose :
            print("Spectrum: {0} DATE={1} Sci_RV={2:.5f} km/s RV_DRIFT={3:.5f} km/s".format(os.path.basename(spectra['filenames'][i]), sci_ccf["header"]["DATE"], sci_ccf["header"]['RV_OBJ'], sci_ccf["header"]["RV_DRIFT"]))
            
        if plot_ccf :
            if i == spectra['nspectra'] - 1 :
                scilegend = "{}".format(sci_ccf["header"]["OBJECT"].replace(" ",""))
            else :
                scilegend = None
            #plt.plot(esci_ccf['RV_CCF'],sci_ccf['MEAN_CCF']-esci_ccf['MEAN_CCF'], "--", label="spectrum")
            plt.plot(sci_ccf['RV_CCF'], sci_ccf['MEAN_CCF'], "-", color='#2ca02c', alpha=0.5, label=scilegend, zorder=1)
        
        if save_spectra :
            output_spectrum_file = spectra['filenames'][i].replace(".fits","_new.fits")
            if verbose :
                print("Saving spectrum to file: {0} ".format(output_spectrum_file))
            sophielib.write_spectrum_to_fits(waves_sf, fluxes, template["fluxerr"], output_spectrum_file, header=header)

    mean_fwhm = np.array(mean_fwhm)
    velocity_window = 1.5*np.nanmedian(mean_fwhm)

    if plot_ccf :
        plt.xlabel('Velocity [km/s]')
        plt.ylabel('CCF')
        plt.legend()
        plt.show()

        calib_rv, median_rv = np.array(calib_rv), np.nanmedian(calib_rv)
        plt.plot(spectra["bjds"], (calib_rv  - median_rv), 'o', color='#2ca02c', label="Sci RV = {0:.4f} km/s".format(median_rv))
        plt.plot(spectra["bjds"], (mean_fwhm  - np.nanmean(mean_fwhm)), '--', color='#2ca02c', label="Sci FWHM = {0:.4f} km/s".format(np.nanmean(mean_fwhm)))
        
        drift_rv = np.array(drift_rv)
        
        mean_drift, sigma_drift = np.nanmedian(drift_rv), np.nanstd(drift_rv)
        plt.plot(spectra["bjds"], drift_rv, '.', color='#ff7f0e', label="Inst. FP drift = {0:.4f}+/-{1:.4f} km/s".format(mean_drift,sigma_drift))

        plt.xlabel(r"BJD")
        plt.ylabel(r"Velocity [km/s]")
        plt.legend()
        plt.show()

    loc = {}

    loc["array_of_spectra"] = array_of_spectra
    loc["spectra"] = spectra
    loc["template"] = template

    loc["ccf_params"] = ccf_params
    loc["ccfmask"] = ccfmask
    
    #loc["fluxkey"], loc["fluxerrkey"] = "sf_fluxes", "sf_fluxerrs"
    #loc["waveskey"], loc["wavekey"] =  "aligned_waves", "common_wl"
    return loc
