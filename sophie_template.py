# -*- coding: iso-8859-1 -*-
"""
    Created on Nov 18 2022
    
    Description: Calculate template and calibrate the time series of SOPHIE spectra, save them into a template product
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python /Volumes/Samsung_T5/Science/sophie/sophie_template.py --input=SOPHIE.202*s1d_A.fits --rv_file=TOI-1736_sophie_ccf.rdb --output=TOI-1736_template.fits -v
    python /Volumes/Samsung_T5/Science/sophie/sophie_template.py --input=SOPHIE.202*s1d_A.fits --rv_file=TOI-2141_sophie_ccf.rdb --output=TOI-2141_template.fits -v
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


sophie_ccf_dir = os.path.dirname(__file__)

parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="Spectral *s1d_A.fits data pattern",type='string',default="*.fits")
parser.add_option("-o", "--output", dest="output", help="Output time series",type='string',default="")
parser.add_option("-r", "--rv_file", dest="rv_file", help="Input file with RVs (km/s)",type='string',default="")
parser.add_option("-a", "--vel_sampling", dest="vel_sampling", help="Velocity sampling for the template spectrum (km/s)",type='float',default=0.5)
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

# First load spectra into a container
array_of_spectra = reduc_lib.load_array_of_sophie_spectra(inputdata, rvfile=options.rv_file, apply_berv=False, plot=False, verbose=options.verbose)
# Then load data into a vector
spectra = reduc_lib.get_spectral_data(array_of_spectra, verbose=options.verbose)
# Set a common wavelength grid for all input spectra
spectra = reduc_lib.set_common_wl_grid(spectra, vel_sampling=options.vel_sampling, verbose=options.verbose)
# Interpolate all spectra to a common wavelength grid
spectra = reduc_lib.resample_and_align_spectra(spectra, verbose=options.verbose, plot=False)
#The following dict entries contain the spectra: spectra["aligned_waves"], spectra["sf_fluxes"],spectra["sf_fluxerrs"], spectra["rest_fluxes"], spectra["rest_fluxerrs"]
# Reduce spectra (template matching, sigma clip, etc.) and generate template
spectra, template = reduc_lib.reduce_spectra(spectra, nsig_clip=5.0, combine_by_median=True, subtract=True, fluxkey="sf_fluxes", fluxerrkey="sf_fluxes", wavekey="common_wl", update_spectra=True, plot=False, verbose=options.verbose)
# Detect continuum and normalize spectra
spectra, template = reduc_lib.normalize_spectra(spectra, template, fluxkey="sf_fluxes", fluxerrkey="sf_fluxes", cont_function='spline3', polyn_order=40, med_filt=1, plot=False)
# Calculate statistical weights based on the time series dispersion 1/sig^2
#spectra = reduc_lib.calculate_weights(spectra, template, use_err_model=False, plot=False)

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
