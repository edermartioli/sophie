# -*- coding: iso-8859-1 -*-
"""
    Created on Nov 18 2022
    
    Description: Calculate template and calibrate the time series of SOPHIE spectra, save them into a template product
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python -W"ignore" /Volumes/Samsung_T5/Science/sophie/sophie_template.py --input=SOPHIE.202*e2ds_A.fits --rv_file=TOI-1736_sophie_ccfrv.rdb --output=TOI-1736_sophie_template.fits -vn
    python -W"ignore" /Volumes/Samsung_T5/Science/sophie/sophie_template.py --input=SOPHIE.202*e2ds_A.fits --rv_file=TOI-2141_sophie_ccfrv.rdb --output=TOI-2141_sophie_template.fits -vn
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
import spectrallib, sophielib
from copy import deepcopy

parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="Spectral *e2ds_A.fits data pattern",type='string',default="*e2ds_A.fits")
parser.add_option("-o", "--output", dest="output", help="Output time series",type='string',default="")
parser.add_option("-r", "--rv_file", dest="rv_file", help="Input file with RVs (km/s)",type='string',default="")
parser.add_option("-a", "--vel_sampling", dest="vel_sampling", help="Velocity sampling for the template spectrum (km/s)",type='float',default=0.5)
parser.add_option("-n", action="store_true", dest="normalize", help="Normalize spectra to the continuum", default=False)
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h sophie_spectral_analysis.py")
    sys.exit(1)

if options.verbose:
    print('Spectral s1d_A.fits data pattern: ', options.input)
    if options.output != "":
        print('Output template product: ', options.output)
    if options.rv_file != "" :
        print('Input file with RVs (km/s) ', options.rv_file)
    print('Velocity sampling (km/s): ', options.vel_sampling)

# make list of spectral data files
if options.verbose:
    print("Creating list of s1d fits spectrum files...")
inputdata = sorted(glob.glob(options.input))


vel_sampling = options.vel_sampling
verbose = options.verbose

spectra = None

headers = reduc_lib.get_headers(inputdata)

norders = 39
for order in range(norders) :

    if options.verbose :
        print("Processing order {} of {} ".format(order+1, norders))
    # First load spectra into a container
    array_of_order_spectra = reduc_lib.load_array_of_sophie_e2ds_spectra(inputdata, order=order, rvfile=options.rv_file, apply_berv=True, silent=True, obslog="", plot=False, verbose=False)

    # Then load data into vector
    order_spectra = reduc_lib.get_spectral_data(array_of_order_spectra, verbose=False)

    # Set a common wavelength grid for all input spectra
    order_spectra = reduc_lib.set_common_wl_grid(order_spectra, vel_sampling=vel_sampling, verbose=False)

    # Interpolate all spectra to a common wavelength grid
    order_spectra = reduc_lib.resample_and_align_spectra(order_spectra, verbose=False, plot=False)
    #order_spectra["aligned_waves"]
    #order_spectra["sf_fluxes"],spectra["sf_fluxerrs"]
    #order_spectra["rest_fluxes"], spectra["rest_fluxerrs"]

    order_spectra, order_template = reduc_lib.reduce_spectra(order_spectra, nsig_clip=5.0, combine_by_median=True, subtract=True, fluxkey="sf_fluxes", fluxerrkey="sf_fluxerrs", wavekey="common_wl", update_spectra=True, plot=False, verbose=False)

    if options.normalize :
        #order_spectra, template = reduc_lib.normalize_spectra(order_spectra, order_template, fluxkey="rest_fluxes", fluxerrkey="rest_fluxerrs", cont_function='spline3', polyn_order=40, med_filt=1, plot=True)
        order_spectra, order_template = reduc_lib.normalize_spectra(order_spectra, order_template, fluxkey="sf_fluxes", fluxerrkey="sf_fluxerrs", cont_function='polynomial', polyn_order=4, plot=False)

    if order == 0 :
        spectra = order_spectra
    else :
        spectra = reduc_lib.append_order(spectra, deepcopy(order_spectra), wave_knot=sophielib.sophie_order_limits()[order])

    del order_spectra
    del array_of_order_spectra
    

# Calculate template for all orders merged together
spectra, template = reduc_lib.reduce_spectra(spectra, nsig_clip=5.0, combine_by_median=True, subtract=True, fluxkey="sf_fluxes", fluxerrkey="sf_fluxerrs", wavekey="common_wl", update_spectra=True, plot=False, verbose=True)

if options.plot :
    plt.plot(template["wl"], template["flux"],'r-', zorder=2)

fluxerrs = []
for j in range(len(template["flux_arr"])) :
    if options.verbose :
        print("Calculating errors for spectrum {} of {}".format(j+1,len(template["flux_arr"])))
    
    ferr = spectrallib.spectral_errors_from_residuals(template["wl"], template["flux_residuals"][j], max_delta_wl=0.2, halfbinsize=5, min_points_per_bin=3, use_mad=True)
    fluxerrs.append(ferr)
    
    if options.plot :
        plt.errorbar(template["wl"], template["flux_arr"][j], yerr=ferr, fmt='.', alpha=0.1, zorder=1)

if options.plot :
    plt.show()

spectrallib.write_spectra_times_series_to_fits(options.output, template["wl"], template["flux"], template["fluxerr"], spectra["bjds"], template["flux_arr"], fluxerrs, header=spectra["header"][0])
