# -*- coding: iso-8859-1 -*-
"""
    Created on April 3 2023
    
    Description: Prepare solar spectrum
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python /Volumes/Samsung_T5/Science/sophie/sophie_prepare_solar_spectrum.py --input=SOPHIE.202*A.fits --rv_file=moon_sophie_ccfrv.rdb --output=solar_spectrum_A.fits -vnp
    python /Volumes/Samsung_T5/Science/sophie/sophie_prepare_solar_spectrum.py --input=SOPHIE.202*B.fits --rv_file=moon_sophie_ccfrv.rdb --output=solar_spectrum_B.fits -vnp
    
    python /Volumes/Samsung_T5/Science/sophie/sophie_prepare_solar_spectrum.py --input=SOPHIE.202*.fits --rv_file=moon_sophie_ccfrv_all.rdb --output=solar_spectrum.fits -vp
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys
import glob

import matplotlib.pyplot as plt
import reduc_lib
import spectrallib

import numpy as np
from copy import deepcopy

parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="Spectral *s1d_A.fits data pattern",type='string',default="*.fits")
parser.add_option("-o", "--output", dest="output", help="Output time series",type='string',default="")
parser.add_option("-r", "--rv_file", dest="rv_file", help="Input file with RVs (km/s)",type='string',default="")
parser.add_option("-a", "--vel_sampling", dest="vel_sampling", help="Velocity sampling for the output spectrum (km/s)",type='float',default=0.5)
parser.add_option("-n", action="store_true", dest="normalize", help="Normalize spectra to the continuum", default=False)
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h sophie_prepare_solar_spectrum.py")
    sys.exit(1)

if options.verbose:
    print('Spectral s1d_A.fits data pattern: ', options.input)
    if options.output != "":
        print('Output spectrum product: ', options.output)
    if options.rv_file != "" :
        print('Input file with RVs (km/s) ', options.rv_file)
    print('Velocity sampling (km/s): ', options.vel_sampling)

# make list of spectral data files
if options.verbose:
    print("Creating list of s1d fits spectrum files...")
inputdata = sorted(glob.glob(options.input))

# First load spectra into a container
array_of_spectra = reduc_lib.load_array_of_sophie_spectra(inputdata, rvfile=options.rv_file, apply_berv=False, plot=False, verbose=options.verbose)
# Then load data into a vector
spectra = reduc_lib.get_spectral_data(array_of_spectra, verbose=options.verbose)
# Set a common wavelength grid for all input spectra
spectra = reduc_lib.set_common_wl_grid(spectra, vel_sampling=options.vel_sampling, verbose=options.verbose)
# Interpolate all spectra to a common wavelength grid
spectra = reduc_lib.resample_and_align_spectra(spectra, verbose=options.verbose, plot=False)

fluxkey="sf_fluxes"
fluxerrkey="sf_fluxes"
wavekey="common_wl"


"""
#cont_function='spline3'
#polyn_order = 80
cont_function='polynomial'
polyn_order = 0

plot_normalization = True

for i in range(spectra['nspectra']) :
    wl = spectra[wavekey]
    flux = spectra[fluxkey][i]
    fluxerr = spectra[fluxerrkey][i]

    keep = np.isfinite(flux)
    keep &= np.isfinite(wl)

    continuum = np.full_like(wl, np.nan)

    if len(flux[keep]) > 10 :
        continuum[keep] = reduc_lib.fit_continuum(wl[keep], flux[keep], function=cont_function, order=polyn_order, nit=10, rej_low=1.5, rej_high=3.5, grow=1, med_filt=False, percentile_low=0., percentile_high=100.,min_points=100, xlabel="wavelength", ylabel="flux", plot_fit=False, silent=True)
        
    if plot_normalization :
        #plt.errorbar(wl, flux, yerr=fluxerr, fmt='.', lw=0.3, alpha=0.3, zorder=1, label="spectral data")
        plt.scatter(wl, flux, marker='o', s=10., edgecolors='tab:blue', facecolors='none', lw=.5)
        plt.plot(wl, continuum, '-', color='brown', lw=2, label="continuum", zorder=2)
    
    for i in range(spectra['nspectra']) :
        spectra[fluxkey][i] /= continuum
        spectra[fluxerrkey][i] /= continuum
        
    if plot_normalization:
        plt.legend(fontsize=18)
        plt.xlabel(r"$\lambda$ [nm]", fontsize=20)
        plt.ylabel(r"Flux", fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.show()
"""

#The following dict entries contain the spectra: spectra["aligned_waves"], spectra["sf_fluxes"],spectra["sf_fluxerrs"], spectra["rest_fluxes"], spectra["rest_fluxerrs"]
# Reduce spectra (template matching, sigma clip, etc.) and generate template
spectra, template = reduc_lib.reduce_spectra(spectra, nsig_clip=0., combine_by_median=True, subtract=True, fluxkey=fluxkey, fluxerrkey=fluxerrkey, wavekey=wavekey, update_spectra=False, plot=False, verbose=options.verbose)

"""
if options.normalize :
    # Detect continuum and normalize spectra
    spectra, template = reduc_lib.normalize_spectra(spectra, template, fluxkey="sf_fluxes", fluxerrkey="sf_fluxes", cont_function='spline3', polyn_order=40, med_filt=1, plot=False)

if options.plot :
    plt.plot(template["wl"], template["flux"],'r-', zorder=2)
"""
fluxerrs = []
for j in range(len(template["flux_arr"])) :
    if options.verbose :
        print("Calculating errors for spectrum {} of {}".format(j+1,len(template["flux_arr"])))
    
    ferr = spectrallib.spectral_errors_from_residuals(template["wl"], template["flux_residuals"][j], max_delta_wl=0.2, halfbinsize=5, min_points_per_bin=3, use_mad=True)
    fluxerrs.append(ferr)
    
    if options.plot :
        plt.errorbar(template["wl"], template["flux_arr"][j], yerr=ferr, fmt='.', alpha=0.1, zorder=1)


use_weighted_mean = True

if use_weighted_mean :

    flux = deepcopy(template["flux"])*0.
    sumweights = np.zeros_like(template["flux"])

    for i in range(len(spectra["bjds"])) :
    
        weights = 1./(fluxerrs[i]*fluxerrs[i])
        #flux += template["flux_arr"][i] * weights
        flux += template["flux_arr"][i]
        sumweights += weights
    
    #flux /= sumweights
    fluxerr = np.sqrt(1./sumweights)
    
    if options.plot :
        plt.plot(template["wl"], flux,'-', color='b')
        plt.errorbar(template["wl"], flux, yerr=fluxerr, fmt='.', color='b')
    #plt.show()
else :
    flux = template["flux"]
    fluxerr = template["fluxerr"]

if options.plot :
    plt.show()

if options.output != "" :
    spectrallib.write_spectra_times_series_to_fits(options.output, template["wl"], flux, fluxerr, spectra["bjds"], template["flux_arr"], fluxerrs, header=spectra["header"][0])

