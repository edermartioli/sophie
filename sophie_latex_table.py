# -*- coding: iso-8859-1 -*-
"""
    Created on Mar 22 2023
    
    Description: Script to make latex table
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python /Volumes/Samsung_T5/Science/sophie/sophie_latex_table.py --input=TOI-1736_sophie_results.txt --input_drs_table=/Volumes/Samsung_T5/Science/TOI-1736/time-series/TOI-1736_spec_ts.txt  --output_latex_file=TOI-1736_sophie_results.tex

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
from astropy.table import Table, hstack
from copy import deepcopy

parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="Input results file name",type='string',default="")
parser.add_option("-d", "--input_drs_table", dest="input_drs_table", help="Input DRS results file name",type='string',default="")
parser.add_option("-x", "--output_latex_file", dest="output_latex_file", help="Output latex file name",type='string',default="")
parser.add_option("-s", "--min_snr", dest="min_snr", help="Minimum SNR",type='float',default=30)
parser.add_option("-m", "--max_moon_c", dest="max_moon_c", help="Maximum Moon contamination",type='float',default=30)
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h sophie_compile_products.py")
    sys.exit(1)


if options.input_drs_table != "" :
    drs_results = ascii.read(options.input_drs_table, data_start=1)

results = ascii.read(options.input, data_start=1)
        
latextbl = Table()
    
#Obs  & Time & RV & $\sigma_{\rm RV}$ & BERV & Exptime & S/N & Fiber-B & FWHM & Bis & QC \\

obs_index = np.arange(len(results['bjd']))

bjd, rv, erv, berv, etime, snr = [], [], [], [], [], []

fwhm, efwhm = [], []
bis, ebis = [], []
sindex, esindex = [], []
ha, eha = [], []
qc = []
fibB_cont = []

for i in obs_index :

    moon_flag, snr_flag, qc_flag = "", "", "OK"

    bjd.append("{:.5f}".format(results['bjd'][i]))
    rv.append("{:.4f}".format(results['vrad'][i]))
    erv.append("{:.4f}".format(results['svrad'][i]))
    berv.append("{:.4f}".format(results['berv'][i]))
    etime.append("{:.0f}".format(results['exptime'][i]))
    snr.append("{:.0f}".format(results['snr'][i]))
    
    fwhm.append("{:.3f}".format(results['fwhm'][i]))
    efwhm.append("{:.3f}".format(results['fwhmerr'][i]))
    bis.append("{:.3f}".format(results['biss'][i]))
    ebis.append("{:.3f}".format(results['bisserr'][i]))
    sindex.append("{:.3f}".format(results['sindex'][i]))
    esindex.append("{:.3f}".format(results['sindexerr'][i]))
    ha.append("{:.3f}".format(results['H-alpha'][i]))
    eha.append("{:.3f}".format(results['H-alphaerr'][i]))


    if options.input_drs_table != "" :
        tdiff = np.abs(drs_results['bjd'] - (results['bjd'][i] - 2400000.))
        match = np.argmin(tdiff)
        
        if tdiff[match] < 0.00001 :
            fibB_cont.append("{:.1f}".format(drs_results["fibB_cont"][match]))
            if drs_results["fibB_cont"][match] > options.max_moon_c :
                moon_flag = "MOON"
        else :
            fibB_cont.append("-")

    if results['snr'][i] < options.min_snr :
        snr_flag = "LOWSNR"

    if snr_flag != "" or moon_flag != "" :
        qc_flag = "{}{}".format(snr_flag, moon_flag)

    qc.append(qc_flag)


latextbl['Obs_index'] = obs_index + 1
latextbl["Time"] = bjd
latextbl["RV"] = rv
latextbl["sigma_RV"] = erv
latextbl["BERV"] = berv
latextbl["Exptime"] = etime
latextbl["SNR"] = snr
if options.input_drs_table != "" :
    latextbl["Fiber-B"] = fibB_cont
latextbl["FWHM"] = fwhm
latextbl["sigma_FWHM"] = efwhm
latextbl["Bis"] = bis
latextbl["sigma_Bis"] = ebis
latextbl["Sindex"] = sindex
latextbl["sigma_Sindex"] = esindex
latextbl["Halpha"] = ha
latextbl["sigma_Halpha"] = eha

#latextbl["QC"] = qc

if options.output_latex_file != "" :
    latextbl.write(options.output_latex_file, format='latex', overwrite=True)
