"""
    Created on Nov 17 2022
    
    Description: This routine performs an analysis of the time-series
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage examples:
    
    python time_series_analysis.py --input=/Volumes/Samsung_T5/Science/TOI-1736/time-series/TOI-1736_spec_ts.txt --period=7.0731267 -pv
    python time_series_analysis.py --input=/Volumes/Samsung_T5/Science/TOI-2141/time-series/TOI-2141_spec_ts.txt --period=18.258946986954 -pv


    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import os, sys

from optparse import OptionParser

import numpy as np
import matplotlib.pyplot as plt

import timeseries_lib as tslib
from astropy.io import ascii

time_series_analysis_dir = os.path.dirname(__file__)


def print_log(filename, snr_threshold=30, moonlight_threshold=0.1) :

    data = ascii.read(options.input)
    
    for i in range(len(data['bjd'])) :
        #isodate = data["dpr_type"][i].split(".")[1]+"."+data["dpr_type"][i].split(".")[2][:3]

        flag = 'OK'
        if data['sn35'][i] <= snr_threshold and data['fibB_cont'][i] <= moonlight_threshold :
            flag = 'LOWSN'
        elif data['sn35'][i] <= snr_threshold and data['fibB_cont'][i] > moonlight_threshold :
            flag = 'LOWSN+MOON'
        elif data['sn35'][i] > snr_threshold and data['fibB_cont'][i] > moonlight_threshold :
            flag = 'MOON'

        #print("{} & {:.5f} & {:.4f} & {:.4f} & {:.4f} & {:.0f} & {:.0f} & {:.1f} & {:.2f} & {:.3f} & {}\\\\".format(i+1, data['bjd'][i]+2400000.0, data['vrad'][i], data['svrad'][i], data['berv'][i], data['Texp'][i], data['sn35'][i], data['fibB_cont'][i], data['fwhm'][i], data['biss'][i], flag))
        #if 'LOWSN' in flag :
        print("{} & {} & {:.5f} & {:.4f} & {:.4f} & {:.4f} & {:.0f} & {:.0f} & {:.1f} & {:.2f} & {:.3f} & {}\\\\".format(i+1, data['dpr_type'][i], data['bjd'][i]+2400000.0, data['vrad'][i], data['svrad'][i], data['berv'][i], data['Texp'][i], data['sn35'][i], data['fibB_cont'][i], data['fwhm'][i], data['biss'][i], flag))


parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help='Input data file',type='string',default="")
parser.add_option("-o", "--period", dest="period", help='Period of expected signal',type='float',default=0.)
parser.add_option("-p", action="store_true", dest="plot", help="plot",default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose",default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with gls_analysis.py -h "); sys.exit(1);

if options.verbose:
    print('Input data file: ', options.input)
    print('Period of expected signal: ', options.period)

##########################################
### LOAD input data
##########################################
if options.verbose:
    print("Loading time series data ...")
    
snr_threshold = 30
moonlight_threshold = 0.1
    
print_log(options.input, snr_threshold=snr_threshold, moonlight_threshold=moonlight_threshold)

timeseriesdata = ascii.read(options.input)
#print(timeseriesdata)

sn = timeseriesdata["sn35"]
moon = timeseriesdata["fibB_cont"]

keep = sn > snr_threshold
keep &= moon <= moonlight_threshold

time, y, yerr = timeseriesdata["bjd"], timeseriesdata["vrad"], timeseriesdata["svrad"]

for i in range(len(time)) :
    if sn[i] > snr_threshold and moon[i] <= moonlight_threshold :
        print("{:.5f}\t{:.5f}\t{:.5f}".format(time[i], y[i], yerr[i]))
    else :
        print("#{:.5f}\t{:.5f}\t{:.5f}".format(time[i], y[i], yerr[i]))


exit()
glsperiodogram = tslib.periodogram(time, y, yerr, nyquist_factor=20, probabilities = [0.01, 0.001], y_label="RV [km/s]",check_period=options.period, npeaks=1, phaseplot=options.plot, plot=options.plot, plot_frequencies=False)

time, y, yerr = timeseriesdata["bjd"], timeseriesdata["fwhm"], timeseriesdata["fwhm"]*0.15
glsperiodogram = tslib.periodogram(time, y, yerr, nyquist_factor=20, probabilities = [0.01, 0.001], y_label="FWHM [km/s]", npeaks=3, phaseplot=options.plot, plot=options.plot, plot_frequencies=False)

time, y, yerr = timeseriesdata["bjd"], timeseriesdata["biss"], timeseriesdata["biss"]*0.15
glsperiodogram = tslib.periodogram(time, y, yerr, nyquist_factor=20, probabilities = [0.01, 0.001], y_label="Bisector slope", npeaks=3, phaseplot=options.plot, plot=options.plot, plot_frequencies=False)



