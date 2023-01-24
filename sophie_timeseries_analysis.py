# -*- coding: iso-8859-1 -*-
"""
    Created on Jan 20 2023
    
    Description: Script to perform a time series analysis of all spectral quantities
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python /Volumes/Samsung_T5/Science/sophie/sophie_timeseries_analysis.py --input=TOI-1736_results.txt -vp

    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii
import corner
#from balrogo import marginals

parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="Input file name",type='string',default="")
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h sophie_timeseries_analysis.py")
    sys.exit(1)


specdata = ascii.read(options.input, data_start=1)
bjd = np.array(specdata['bjd'])
ns = len(bjd)

#select_cols = ['snr','berv','airmass','vrad','biss','fwhm','sindex','H-alpha','NaI','CaI']
select_cols = ['berv','vrad','biss','fwhm','sindex','H-alpha','NaI','CaI']

slabels, labels = [], []
for col in specdata.columns :
    labels.append(col)
    if col in select_cols :
        slabels.append(col)
    
samples = []
for i in range(ns) :
    samples.append([])
    for j in range(len(slabels)) :
        samples[i].append(np.array(specdata[slabels[j]])[i])
samples = np.array(samples, dtype=float)

#fig = marginals.corner(samples,labels=slabels,quantiles=[0.16, 0.5, 0.84])
fig = corner.corner(samples, show_titles=True, labels = newlabels, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84], truths=truths, labelsize=12, labelpad=2.0)

for ax in fig.get_axes():
    plt.setp(ax.get_xticklabels(), ha="left", rotation=45)
    plt.setp(ax.get_yticklabels(), ha="right", rotation=45)
    ax.tick_params(axis='both', labelsize=8)

plt.show()
