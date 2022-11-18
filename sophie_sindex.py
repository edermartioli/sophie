# -*- coding: iso-8859-1 -*-
"""
    Created on Nov 11 2022
    
    Description: Caculate s-index for a template and time series of SOPHIE spectra
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python /Volumes/Samsung_T5/Science/sophie/sophie_sindex.py --input=TOI-2141_template.fits -pv
    python /Volumes/Samsung_T5/Science/sophie/sophie_sindex.py --input=TOI-1736_template.fits -pv

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

sophie_ccf_dir = os.path.dirname(__file__)

parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="Spectral *s1d_A.fits data pattern",type='string',default="*.fits")
parser.add_option("-o", "--output", dest="output", help="Output s-index time series",type='string',default="")
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
        print('Output s-index time series: ', options.output)

# Loading input template product
if options.verbose:
    print("Loading input template product ...")

template = spectrallib.read_template_product(options.input)

##################################
#### START SPECTRAL ANALYS #######
##################################
# extract fluxes within a certain spectral window and re-normalize data by a local continuum
wl, flux, fluxerr, fluxes, fluxerrs = spectrallib.extract_spectral_feature(template, wlrange=[388.8,402.0], cont_ranges=[[388.8,391.5],[394.6379,395.6379],[399.0,402.0]], polyn_order=6, plot=options.plot)

# run the routine sindex() to estimate the S-index from the template spectrum and to generate some pretty plots
template_sindex = spectrallib.sindex(wl, flux, deltalamca=0.1, deltalamcont=1.0, lamh=393.368, lamk=396.849, lamv=390.107, lamr=400.107, verbose=False, plot=options.plot)

exit()

# run the s-index routine with Monte Carlo to obtain the posterior probability distributino for the values of the sindex
template_sindex, template_sindexerr = spectrallib.sindex_montecarlo(wl, flux, fluxerr, deltalamca=0.1, deltalamcont=1.0, lamh=393.368, lamk=396.849, lamv=390.107, lamr=400.107, verbose=True, plot=options.plot)

# Run the sindex MC routine for all individual spectra to obtain the time series
sindex, sindexerr = np.array([]), np.array([])
for i in range(len(fluxes)) :
    #print("Calculating S-index for spectrum {} of {}".format(i+1,len(fluxes)))
    sidx, esidx = spectrallib.sindex_montecarlo(wl, fluxes[i], fluxerrs[i], deltalamca=0.1, deltalamcont=1.0, lamh=393.368, lamk=396.849, lamv=390.107, lamr=400.107, verbose=False, plot=False)
    sindex = np.append(sindex,sidx)
    sindexerr = np.append(sindexerr,esidx)

    if options.verbose :
        print("{:.8f} {:.4f} {:.4f}".format(template["times"][i], sidx, esidx))

# Save s-index time series to output file
if options.output != "":
    spectrallib.save_time_series(options.output, template["times"], sindex, sindexerr)

# Plot time series
if options.plot :
    plt.errorbar(template["times"], sindex, yerr=sindexerr, fmt='o', color='k')
    plt.hlines(template_sindex, template["times"][0], template["times"][-1], ls="-", lw=3, color="darkgreen", label="Template S-index = {:.4f}+/-{:.4f}".format(template_sindex,template_sindexerr))
    plt.hlines(template_sindex+template_sindexerr, template["times"][0], template["times"][-1], ls="--", color="darkgreen")
    plt.hlines(template_sindex-template_sindexerr, template["times"][0], template["times"][-1], ls="--", color="darkgreen")
    plt.xlabel(r"BJD", fontsize=20)
    plt.ylabel(r"S-index", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    plt.show()
