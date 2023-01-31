"""
    Created on Nov 17 2022
    
    Description: library to calculate spectral quantities
    
    @author: Eder Martioli <emartioli@lna.br>, <martioli@iap.fr>
    
    Laboratorio Nacional de Astrofisica, Brazil
    Institut d'Astrophysique de Paris, France
    
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import os,sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import scipy.signal as signal
import scipy.interpolate as sint
import warnings
from scipy import stats

from astropy.io import fits

from copy import deepcopy

def spectral_errors_from_residuals(wl, residuals, max_delta_wl=0.2, halfbinsize=15, min_points_per_bin=3, use_mad=True) :

    npoints = len(residuals)

    arrays = []
    
    # add central array
    arrays.append(residuals)
    
    for i in range(1,halfbinsize) :
    
        # initialize arrays 
        tmpbefore, tmpafter = np.full_like(residuals,np.nan), np.full_like(residuals,np.nan)
        
        # add array shifted by i pixels forward
        tmpbefore[:-i] = residuals[i:]
        arrays.append(tmpbefore)

        # add array shifted by i pixels backwards
        tmpafter[i:] = residuals[:-i]
        arrays.append(tmpafter)

    arrays = np.array(arrays,dtype=float)

    if use_mad :
        errors = stats.median_absolute_deviation(arrays, axis=0, nan_policy='omit')
    else :
        errors = stats.median_absolute_deviation(arrays, axis=0, nan_policy='omit')
    
    """
    The algorithm below is slower, so I kept only the above one, but it's a better option since it uses the wl information
    for i in range(
        arrays.append(np.full_like(residuals))

    for i in range(npoints) :
        i1, i2 = 0, len(wl)
        if (i - halfbinsize) >= 0 :
            i1 = i - halfbinsize
        if (i + halfbinsize) <= npoints :
            i2 = i + halfbinsize
        
        wlchunk = wl[i1:i2]
        residchunk = residuals[i1:i2]
        
        wldiff = np.abs(wlchunk - wl[i])
        keep = wldiff < max_delta_wl
        
        if len(residchunk[keep]) >= min_points_per_bin :
            if use_mad :
                errors[i] = stats.median_absolute_deviation(residchunk[keep],nan_policy='omit')
            else :
                errors[i] = np.nanstd(residchunk[keep])
        else :
            errors[i] = residuals[i] / 0.67449
     """
    return errors

def delta_mag(flux_nan):
    delta_mag = 2*(np.sqrt(2)*np.sqrt(np.mean(np.square(flux_nan))))
    return delta_mag

def angle_check(lamb,flux,lamv,deltalamcont):
    model = LinearRegression()
    hlinex0 = lamb[lamb < (lamv+deltalamcont)]
    hliney0 = flux[lamb < (lamv+deltalamcont)]
    hlinex = hlinex0[hlinex0 > (lamv-deltalamcont)]
    hliney = hliney0[hlinex0 > (lamv-deltalamcont)]
    x = hlinex.reshape((-1, 1))
    y = hliney
    model.fit(x, y)
    model = LinearRegression().fit(x, y)
    y_pred = model.predict(x)
    return x, y_pred, model.coef_


def extract_spectral_feature(template, wlrange=[], cont_ranges=[], polyn_order=6, divide_by_deltamag=False, plot=False) :
    
    wl = template["wl"]
    flux = template["flux"]
    fluxerr = template["fluxerr"]

    if wlrange == [] :
        wlrange = [wl[0],wl[-1]]

    keep = (wl > wlrange[0]) & (wl < wlrange[1])
    keep &= (np.isfinite(flux)) & (np.isfinite(fluxerr))
    
    wl, flux, fluxerr = wl[keep], flux[keep], fluxerr[keep]
        
    cont = wl < 0
    for r in cont_ranges :
        cont ^= (wl > r[0]) & (wl < r[-1])

    coeffs = fit_continuum(wl[cont], flux[cont], function='polynomial', order=polyn_order, nit=5, rej_low=1., rej_high=4.0, grow=1, med_filt=1, percentile_low=0., percentile_high=100., min_points=100, xlabel="wavelength", ylabel="flux", return_polycoeffs=True, plot_fit=False, verbose=False)

    continuum = np.poly1d(coeffs)(wl)

    if plot :
        #plt.errorbar(wl, flux, yerr=fluxerr, fmt='.', lw=0.3, alpha=0.3, zorder=1, label="template spectrum")
        plt.plot(wl, flux, '-', lw=2, alpha=0.8, zorder=1, label="template spectrum")
        plt.plot(wl, continuum, '--', lw=2, zorder=2, label="continuum")

    flux /= continuum
    fluxerr /= continuum
    
    if divide_by_deltamag :
        deltamag_factor =  delta_mag(flux)
        flux /= deltamag_factor
        fluxerr /= deltamag_factor
    
    fluxes, fluxerrs = [], []
    
    for j in range(len(template["fluxes"])) :
        f = template["fluxes"][j][keep] / continuum
        ferr = template["fluxerrs"][j][keep] / continuum
        
        if divide_by_deltamag :
            f /= deltamag_factor
            ferr /= deltamag_factor
            
        fluxes.append(f / continuum)
        fluxerrs.append(ferr / continuum)
        #fluxerrs.append(template["flux_residuals"][j][keep] / continuum)
        if plot :
            if j==0:
                plt.plot(wl,template["fluxes"][j][keep],'k.',alpha=0.1,zorder=0.5, label="spectral data")
            else :
                plt.plot(wl,template["fluxes"][j][keep],'k.',alpha=0.1,zorder=0.5)

    if plot :
        plt.legend(fontsize=18)
        plt.xlabel(r"$\lambda$ [nm]", fontsize=20)
        plt.ylabel(r"Flux", fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.show()

    return wl, flux, fluxerr, fluxes, fluxerrs



def sindex(wl, flux, deltalamca=0.1, deltalamcont=1.0, lamh=393.368, lamk=396.849, lamv=390.107, lamr=400.107, verbose=False, plot=False) :
    
    sindex = np.nan

    """
    check_min = 1.8
    fitX, fitY, conf_degree = angle_check(wl,flux,lamh,deltalamcont/2)
    fitX2, fitY2, conf_degree2 = angle_check(wl,flux,lamh+(lamk-lamh)/2,deltalamcont/2)
    check = 1/(fitY[int(len(fitY)/2)]/fitY2[int(len(fitY2)/2)])
    print("Check value: {:.2f}".format(check))
    if check > check_min :
    """
    
    contr = (wl > lamr-deltalamcont) & (wl < lamr+deltalamcont)
    contv = (wl > lamv-deltalamcont) & (wl < lamv+deltalamcont)

    lineh = (wl > lamh-deltalamca) & (wl < lamh+deltalamca)
    linek = (wl > lamk-deltalamca) & (wl < lamk+deltalamca)

    fluxh = np.nansum(flux[lineh] * (deltalamca - np.abs(wl[lineh] - lamh)))
    triagband_h = deltalamca - np.abs(wl[lineh] - lamh)
    weighth = np.nansum(triagband_h)
    fluxh /= weighth

    fluxk = np.nansum(flux[linek] * (deltalamca - np.abs(wl[linek] - lamk)))
    triagband_k = deltalamca - np.abs(wl[linek] - lamk)
    weightk = np.nansum(triagband_k)
    fluxk /= weightk

    fluxr = np.nanmean(flux[contr])
    fluxv = np.nanmean(flux[contv])

    sindex = (fluxh+fluxk)/(fluxv+fluxr)

    if plot :
        plt.plot(wl,flux,"-",lw=0.7, color="k", zorder=0.5, label="spectrum")

        #plt.errorbar(wl[lineh],flux[lineh],yerr=fluxerr[lineh],fmt=".", color="darkred", alpha=0.7, label="Ca II H")
        plt.plot(wl[lineh],flux[lineh],".", color="darkblue", alpha=0.7)
        plt.fill_between(x=wl[lineh], y1=triagband_h/np.max(triagband_h), y2=np.zeros_like(flux[lineh]), color= "darkblue",alpha= 0.2, label="Ca II H")

        #plt.errorbar(wl[linek],flux[linek],yerr=fluxerr[linek],fmt=".", color="darkblue", alpha=0.7, label="Ca II K")
        plt.plot(wl[linek],flux[linek],".", color="darkred", alpha=0.7)
        plt.fill_between(x=wl[linek], y1=triagband_k/np.max(triagband_k), y2=np.zeros_like(flux[linek]), color= "darkred",alpha= 0.2, label="Ca II K")

        #plt.errorbar(wl[contv],flux[contv],yerr=fluxerr[contv],fmt=".", color="darkgreen", alpha=0.7, label="Continuum v")
        plt.plot(wl[contv],flux[contv],".", color="darkgreen", alpha=0.7)
        plt.fill_between(x=wl[contv], y1=np.ones_like(flux[contv]), y2=np.zeros_like(flux[contv]), color= "darkgreen",alpha=0.2, label="Cont V")

        #plt.errorbar(wl[contr],flux[contr],yerr=fluxerr[contr],fmt=".", color="darkgreen", alpha=0.7, label="Continuum r")
        plt.plot(wl[contr], flux[contr], ".", color="olive", alpha=0.7)
        plt.fill_between(x=wl[contr], y1=np.ones_like(flux[contr]), y2=np.zeros_like(flux[contr]), color= "olive",alpha=0.2, label="Cont R")

        plt.legend(fontsize=16)
        plt.xlabel(r"$\lambda$ [nm]", fontsize=22)
        plt.ylabel(r"Flux", fontsize=22)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.show()
    
    if verbose :
        print("Sindex={:.4f}".format(sindex))
    
    return sindex


def sindex_montecarlo(wl, flux, fluxerr, deltalamca=0.1, deltalamcont=1.0, lamh=393.368, lamk=396.849, lamv=390.107, lamr=400.107, nsamples=1000, verbose=False, plot=False) :
    
    contr = (wl > lamr-deltalamcont) & (wl < lamr+deltalamcont)
    contv = (wl > lamv-deltalamcont) & (wl < lamv+deltalamcont)

    lineh = (wl > lamh-deltalamca) & (wl < lamh+deltalamca)
    linek = (wl > lamk-deltalamca) & (wl < lamk+deltalamca)

    hfactor = deltalamca - np.abs(wl[lineh] - lamh)
    weighth = np.nansum(deltalamca - np.abs(wl[lineh] - lamh))
    
    kfactor = deltalamca - np.abs(wl[linek] - lamk)
    weightk = np.nansum(deltalamca - np.abs(wl[linek] - lamk))

    flux_samples = []
    for i in range(len(flux)) :
        flux_samples.append(np.random.normal(flux[i], fluxerr[i], nsamples))
    flux_samples = np.array(flux_samples, dtype=float)

    sindex_samples = np.array([])
    for s in range(nsamples) :

        fluxh = np.nansum(flux_samples[:,s][lineh] * hfactor) / weighth
        fluxk = np.nansum(flux_samples[:,s][linek] * kfactor) / weightk

        fluxr = np.nanmean(flux_samples[:,s][contr])
        fluxv = np.nanmean(flux_samples[:,s][contv])

        sindex = (fluxh+fluxk)/(fluxv+fluxr)
        
        sindex_samples = np.append(sindex_samples, sindex)

    sindex_percentiles = np.percentile(sindex_samples, [16, 50, 84], axis=0)
    sindex = sindex_percentiles[1]
    sindex_max_err = sindex_percentiles[2]-sindex_percentiles[1]
    sindex_min_err = sindex_percentiles[1]-sindex_percentiles[0]
    sindexerr = (sindex_max_err + sindex_min_err) / 2

    if verbose :
        print("S-index = {0:.3f} + {1:.3f} - {2:.3f} ".format(sindex, sindex_max_err, sindex_min_err))
              
    if plot :
        count, bins, ignored = plt.hist(sindex_samples, 30, density=True)
        plt.plot(bins, 1/(sindexerr * np.sqrt(2 * np.pi)) * np.exp( - (bins - sindex)**2 / (2 * sindexerr**2) ), linewidth=2, color='r')
        plt.xlabel(r"S-index", fontsize=22)
        plt.ylabel(r"Probability density", fontsize=22)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.show()

    return sindex, sindexerr



def activity_index(wl, flux, wl_lines=[656.28], delta_wl_lines=[0.15], wl_conts=[655.087,658.031], delta_wl_conts=[1.075,0.875], line_label="H-alpha", verbose=False, plot=False) :
    
    index_value = np.nan

    cont = np.full_like(wl, False, dtype=bool)
    cont_masks = []
    
    for i in range(len(wl_conts)) :
        contmask = (wl > wl_conts[i]-delta_wl_conts[i]/2) & (wl < wl_conts[i]+delta_wl_conts[i]/2)
        cont_masks.append(contmask)
        cont ^= contmask
        
    line = np.full_like(wl, False, dtype=bool)
    line_masks = []
    for i in range(len(wl_lines)) :
        linemask = (wl > wl_lines[i]-delta_wl_lines[i]/2) & (wl < wl_lines[i]+delta_wl_lines[i]/2)
        line_masks.append(linemask)
        line ^= linemask
        
    fluxline = np.nanmean(flux[line])
    fluxcont = np.nanmean(flux[cont])

    index_value = (fluxline)/(fluxcont)

    if plot :
        plt.plot(wl,flux,"-",lw=0.7, color="k", zorder=0.5, label="spectrum")

        plt.plot(wl[line],flux[line],".", color="darkblue", alpha=0.7, label="{}".format(line_label))
        for i in range(len(wl_lines)) :
            plt.fill_between(x=wl[line_masks[i]], y1=np.ones_like(flux[line_masks[i]]), y2=np.zeros_like(flux[line_masks[i]]), color= "darkblue",alpha= 0.2)

        plt.plot(wl[cont],flux[cont],".", color="darkgreen", alpha=0.7, label="Continuum")
        for i in range(len(wl_conts)) :
            plt.fill_between(x=wl[cont_masks[i]], y1=np.ones_like(flux[cont_masks[i]]), y2=np.zeros_like(flux[cont_masks[i]]), color= "darkgreen",alpha=0.2)

        plt.legend(fontsize=16)
        plt.xlabel(r"$\lambda$ [nm]", fontsize=22)
        plt.ylabel(r"Flux", fontsize=22)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.show()
    
    if verbose :
        print("{} index = {:.4f}".format(line_label,index_value))
    
    return index_value


def activity_index_montecarlo(wl, flux, fluxerr, wl_lines=[656.28], delta_wl_lines=[0.15], wl_conts=[655.087,658.031], delta_wl_conts=[1.075,0.875], nsamples=1000, line_label="H$\alpha$", verbose=False, plot=False) :
    
    cont = np.full_like(wl, False, dtype=bool)
    for i in range(len(wl_conts)) :
        cont ^= (wl > wl_conts[i]-delta_wl_conts[i]/2) & (wl < wl_conts[i]+delta_wl_conts[i]/2)
        
    line = np.full_like(wl, False, dtype=bool)
    for i in range(len(wl_lines)) :
        line ^= (wl > wl_lines[i]-delta_wl_lines[i]/2) & (wl < wl_lines[i]+delta_wl_lines[i]/2)

    flux_samples = []
    for i in range(len(flux)) :
        flux_samples.append(np.random.normal(flux[i], fluxerr[i], nsamples))
    flux_samples = np.array(flux_samples, dtype=float)

    index_samples = np.array([])
    for s in range(nsamples) :

        fluxline = np.nanmean(flux_samples[:,s][line])
        fluxcont = np.nanmean(flux_samples[:,s][cont])

        index_value = fluxline/fluxcont
        
        index_samples = np.append(index_samples, index_value)

    index_percentiles = np.percentile(index_samples, [16, 50, 84], axis=0)
    index_value = index_percentiles[1]
    index_max_err = index_percentiles[2]-index_percentiles[1]
    index_min_err = index_percentiles[1]-index_percentiles[0]
    indexerr = (index_max_err + index_min_err) / 2

    if verbose :
        print("{} index = {:.4f} + {:.4f} - {:.4f} ".format(line_label, index_value, index_max_err, index_min_err))
              
    if plot :
        count, bins, ignored = plt.hist(index_samples, 30, density=True)
        plt.plot(bins, 1/(indexerr * np.sqrt(2 * np.pi)) * np.exp( - (bins - index_value)**2 / (2 * indexerr**2) ), linewidth=2, color='r')
        plt.xlabel(r"{} index".format(line_label), fontsize=22)
        plt.ylabel(r"Probability density", fontsize=22)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.show()

    return index_value, indexerr


def activity_index_timeseries(wl, fluxes, fluxerrs, times, wl_lines=[656.28], delta_wl_lines=[0.15], wl_conts=[655.087,658.031], delta_wl_conts=[1.075,0.875], nsamples=1000, line_label="CaI", output="", ref_index_value=0, ref_index_err=0, verbose=False, plot=False) :

    # Run the MC routine for all individual spectra to obtain the activity index time series
    CaI, CaIerr = np.array([]), np.array([])
    index_values, index_errs = np.array([]), np.array([])
    
    for i in range(len(fluxes)) :
    
        idx, eidx = activity_index_montecarlo(wl, fluxes[i], fluxerrs[i], wl_lines=wl_lines, delta_wl_lines=delta_wl_lines, wl_conts=wl_conts, delta_wl_conts=delta_wl_conts, nsamples=nsamples, line_label=line_label, verbose=False, plot=False)
    
        index_values = np.append(index_values,idx)
        index_errs = np.append(index_errs,eidx)

        if verbose :
            print("Spectrum {}/{} -> BJD = {:.8f} {} index = {:.4f} +/- {:.4f}".format(i+1, len(fluxes), times[i], line_label, idx, eidx))

    # Save s-index time series to output file
    if output != "":
        save_time_series(output, times-2400000., index_values, index_errs, xlabel="rjd", ylabel=line_label, yerrlabel="{}err".format(line_label), write_header_rows=True)

    # Plot time series
    if plot :
        plt.errorbar(times-2400000., index_values, yerr=index_errs, fmt='o', color='k')
        if ref_index_value and ref_index_err :
            plt.hlines(ref_index_value, times[0]-2400000., times[-1]-2400000., ls="-", lw=3, color="darkgreen", label=r"Template {} index = {:.4f}$\pm${:.4f}".format(line_label, ref_index_value,ref_index_err))
            plt.fill_between(x=times-2400000., y1=np.full_like(times-2400000.,ref_index_value+ref_index_err), y2=np.full_like(times-2400000.,ref_index_value-ref_index_err), color= "darkgreen",alpha= 0.3)

        plt.xlabel(r"BJD-2400000", fontsize=20)
        plt.ylabel(r"{} index".format(line_label), fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=18)
        plt.show()


    return times, index_values, index_errs


def save_time_series(output, time, y, yerr, xlabel="x", ylabel="y", yerrlabel="yerr", write_header_rows=False) :
    
    outfile = open(output,"w+")
    
    if write_header_rows :
        outfile.write("{}\t{}\t{}\n".format(xlabel,ylabel,yerrlabel))
        outfile.write("---\t----\t-----\n")
        
    for i in range(len(time)) :
        outfile.write("{:.10f}\t{:.5f}\t{:.5f}\n".format(time[i], y[i], yerr[i]))

    outfile.close()


def save_rv_time_series(output, bjd, rv, rverr, time_in_rjd=True, rv_in_mps=False) :
    
    outfile = open(output,"w+")
    outfile.write("rjd\tvrad\tsvrad\n")
    outfile.write("---\t----\t-----\n")
    
    for i in range(len(bjd)) :
        if time_in_rjd :
            rjd = bjd[i] - 2400000.
        else :
            rjd = bjd[i]
        
        if rv_in_mps :
            outfile.write("{0:.10f}\t{1:.2f}\t{2:.2f}\n".format(rjd, 1000. * rv[i], 1000. * rverr[i]))
        else :
            outfile.write("{0:.10f}\t{1:.5f}\t{2:.5f}\n".format(rjd, rv[i], rverr[i]))

    outfile.close()


def write_spectra_times_series_to_fits(filename, wave, flux, fluxerr, times, fluxes, fluxerrs, header=None):
    """
        Description: function to save the spectrum to a fits file
        """
    
    if header is None :
        header = fits.Header()

    header.set('TTYPE1', "WAVE")
    header.set('TUNIT1', "NM")
    header.set('TTYPE2', "FLUXES")
    header.set('TUNIT2', "COUNTS")
    header.set('TTYPE2', "FLUXERR")
    header.set('TUNIT2', "COUNTS")

    primary_hdu = fits.PrimaryHDU(header=header)
    hdu_wl = fits.ImageHDU(data=wave, name="WAVE")
    hdu_tmpflux = fits.ImageHDU(data=flux, name="TEMPLATE_FLUX")
    hdu_tmpfluxerr = fits.ImageHDU(data=fluxerr, name="TEMPLATE_FLUXERR")

    hdu_times = fits.ImageHDU(data=times, name="TIMES")
    hdu_fluxes = fits.ImageHDU(data=fluxes, name="FLUXES")
    hdu_fluxerrs = fits.ImageHDU(data=fluxerrs, name="FLUXERRS")

    listofhuds = [primary_hdu, hdu_wl, hdu_tmpflux, hdu_tmpfluxerr, hdu_times, hdu_fluxes, hdu_fluxerrs]

    mef_hdu = fits.HDUList(listofhuds)

    mef_hdu.writeto(filename, overwrite=True)


def read_template_product(filename) :

    hdulist = fits.open(filename)

    wave = hdulist["WAVE"].data
    flux = hdulist["TEMPLATE_FLUX"].data
    fluxerr = hdulist["TEMPLATE_FLUXERR"].data
    times = hdulist["TIMES"].data
    fluxes = hdulist["FLUXES"].data
    fluxerrs = hdulist["FLUXERRS"].data

    loc = {}
    loc["wl"] = wave
    loc["flux"] = flux
    loc["fluxerr"] = fluxerr
    loc["times"] = times
    loc["fluxes"] = fluxes
    loc["fluxerrs"] = fluxerrs

    return loc

def fit_continuum(wav, spec, function='polynomial', order=3, nit=5, rej_low=2.0,
    rej_high=2.5, grow=1, med_filt=0, percentile_low=0., percentile_high=100.,
                  min_points=10, xlabel="", ylabel="", return_polycoeffs=False, plot_fit=True, verbose=False):
    
    warnings.simplefilter('ignore', np.RankWarning)
    
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
    mspec = np.ma.masked_array(spec, mask=np.zeros_like(spec))
    # mask 1st and last point: avoid error when no point is masked
    # [not in IRAF]
    mspec.mask[0] = True
    mspec.mask[-1] = True
    
    mspec = np.ma.masked_where(np.isnan(spec), mspec)
    
    # apply median filtering prior to fit
    # [opt] [not in IRAF]
    if int(med_filt):
        fspec = signal.medfilt(spec, kernel_size=med_filt)
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

    if return_polycoeffs :
        #cont = np.poly1d(coeff)(wav)
        return coeff
    else :
        return cont
