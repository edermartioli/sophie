# -*- coding: iso-8859-1 -*-
"""
    Created on Jan 18 2022
    
    Description: Caculate Na I index for a template and time series of SOPHIE spectra
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python /Volumes/Samsung_T5/Science/sophie/sophie_NaI.py --input=TOI-2141_template.fits -pv
    python /Volumes/Samsung_T5/Science/sophie/sophie_NaI.py --input=TOI-1736_template.fits -pv

    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys
import glob

import matplotlib.pyplot as plt
import spectrallib

import numpy as np

parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="Input pectral data",type='string',default="")
parser.add_option("-o", "--output", dest="output", help="Output Na I time series",type='string',default="")
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h sophie_NaI.py")
    sys.exit(1)

if options.verbose:
    print('Input pectral data: ', options.input)
    if options.output != "":
        print('Output NaI time series: ', options.output)

# Loading input template product
if options.verbose:
    print("Loading input template product ...")

# load spectral data from a template product
template = spectrallib.read_template_product(options.input)

##################################
#### SELECT SPECTRAL RANGE #######
##################################
# extract fluxes within a certain spectral window and re-normalize data by a local continuum
wl, flux, fluxerr, fluxes, fluxerrs = spectrallib.extract_spectral_feature(template, wlrange=[579.5,610.5], cont_ranges=[[580.0,581.0],[608.0,610.0]], polyn_order=1, plot=options.plot)

##################################
#### START SPECTRAL ANALYS #######
##################################
line_label="NaI"

wl_lines = [589.592,588.995]
delta_wl_lines = [0.1,0.1]

wl_conts = [580.50,609.00]
delta_wl_conts = [1.0,2.0]

# run the routine to estimate the feature index from the template spectrum and to generate some nice plots
template_halpha = spectrallib.activity_index(wl, flux, wl_lines=wl_lines, delta_wl_lines=delta_wl_lines, wl_conts=wl_conts, delta_wl_conts=delta_wl_conts, line_label=line_label, verbose=options.verbose, plot=options.plot)

# run a routine with Monte Carlo to obtain the posterior probability distribution for the values of the feature index
template_halpha, template_halphaerr = spectrallib.activity_index_montecarlo(wl, flux, fluxerr, wl_lines=wl_lines, delta_wl_lines=delta_wl_lines, wl_conts=wl_conts, delta_wl_conts=delta_wl_conts, nsamples=1000, line_label=line_label, verbose=options.verbose, plot=options.plot)

# Run the MC routine for all individual spectra to obtain the time series
times, halpha, halphaerr = spectrallib.activity_index_timeseries(wl, fluxes, fluxerrs, template["times"], wl_lines=wl_lines, delta_wl_lines=delta_wl_lines, wl_conts=wl_conts, delta_wl_conts=delta_wl_conts, nsamples=1000, line_label=line_label, output=options.output, ref_index_value=template_halpha, ref_index_err=template_halphaerr, verbose=options.verbose, plot=options.plot)
