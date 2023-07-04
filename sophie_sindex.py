# -*- coding: iso-8859-1 -*-
"""
    Created on Nov 11 2022
    
    Description: Caculate s-index for a template and time series of SOPHIE spectra
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python /Volumes/Samsung_T5/Science/sophie/sophie_sindex.py --input=TOI-2141_sophie_template.fits --teff="" --B="10.13,0.05" --V="9.46,0.003"  --teff="5656,7" --feh=-0.16 --mass=0.920 -pv
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
import spectrallib

from uncertainties import ufloat

import numpy as np

parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="Input pectral data",type='string',default="")
parser.add_option("-o", "--output", dest="output", help="Output s-index time series",type='string',default="")
parser.add_option("-k", "--outputRHK", dest="outputRHK", help="Output log(R'HK) time series",type='string',default="")
parser.add_option("-t", "--teff", dest="teff", help="Input Teff, eTeff",type='string',default="5787,8")
parser.add_option("-B", "--B", dest="bmag", help="Input B, eB",type='string',default="9.64,0.03")
parser.add_option("-V", "--V", dest="vmag", help="Input V, eV",type='string',default="8.953,0.002")
parser.add_option("-f", "--feh", dest="feh", help="Input metallicity [Fe/H]",type='float',default=+0.09)
parser.add_option("-m", "--mass", dest="mass", help="Input star mass [Msun]",type='float',default=1.05)
parser.add_option("-r", action="store_true", dest="instrumental", help="instrumental", default=False)
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h sophie_sindex.py")
    sys.exit(1)

if options.verbose:
    print('Input pectral data: ', options.input)
    if options.output != "":
        print('Output s-index time series: ', options.output)

# Loading input template product
if options.verbose:
    print("Loading input template product ...")

template = spectrallib.read_template_product(options.input)

teff = ufloat(5787,8)
bv = ufloat(9.64,0.03) - ufloat(8.953,0.002)
mass = options.mass
feh = options.feh #[Fe/H]

##################################
#### START SPECTRAL ANALYS #######
##################################
# extract fluxes within a certain spectral window and re-normalize data by a local continuum
wl, flux, fluxerr, fluxes, fluxerrs = spectrallib.extract_spectral_feature(template, wlrange=[388.8,402.0], cont_ranges=[[388.8,391.5],[394.6379,395.6379],[399.0,402.0]], normalize=True, polyn_order=1, plot=options.plot)

# run the routine sindex() to estimate the S-index from the template spectrum and to generate some pretty plots
template_sindex = spectrallib.sindex(wl, flux, deltalamca=0.1, deltalamcont=1.0, lamh=393.368, lamk=396.849, lamv=390.107, lamr=400.107, verbose=False, plot=options.plot)

# run the s-index routine with Monte Carlo to obtain the posterior probability distributino for the values of the sindex
template_sindex, template_sindexerr = spectrallib.sindex_montecarlo(wl, flux, fluxerr, deltalamca=0.1, deltalamcont=1.0, lamh=393.368, lamk=396.849, lamv=390.107, lamr=400.107, verbose=True, plot=options.plot)

usindex = ufloat(template_sindex, template_sindexerr)
cal_template_sindex = spectrallib.calibrate_sindex_to_MW(usindex)

ss = spectrallib.SMW_RHK()
logrhkprime = ss.montecarlo_SMWtoRHK(cal_template_sindex, teff, bv, nsamples=1000, lc="ms", verbose=options.verbose, plot=options.plot)

if options.verbose :
    print("SMW = {0:.3f} +/- {1:.3f} ".format(cal_template_sindex.nominal_value, cal_template_sindex.std_dev))
    print("log(R'HK) = {0:.3f} +/- {1:.3f} ".format(logrhkprime.nominal_value, logrhkprime.std_dev))
    print("logR'HK(B-V)=",np.round(spectrallib.MW(cal_template_sindex.nominal_value,bv.nominal_value),2) )
    
    logrlhk = spectrallib.MW(cal_template_sindex.nominal_value,bv.nominal_value)
    spectrallib.MH08(logrlhk, bv.nominal_value)
    spectrallib.LO18(cal_template_sindex.nominal_value, teff.nominal_value)
    spectrallib.LO16(logrlhk, mass, feh)

# Run the sindex MC routine for all individual spectra to obtain the time series
sindex, sindexerr = np.array([]), np.array([])
smw, smwerr = np.array([]), np.array([])
rhk, rhkerr = np.array([]), np.array([])

for i in range(len(fluxes)) :
    #print("Calculating S-index for spectrum {} of {}".format(i+1,len(fluxes)))
    sidx, esidx = spectrallib.sindex_montecarlo(wl, fluxes[i], fluxerrs[i], deltalamca=0.1, deltalamcont=1.0, lamh=393.368, lamk=396.849, lamv=390.107, lamr=400.107, verbose=False, plot=False)
    
    sindex = np.append(sindex,sidx)
    sindexerr = np.append(sindexerr,esidx)

    usindex = ufloat(sidx, esidx)
    cal_sindex = spectrallib.calibrate_sindex_to_MW(usindex)

    smw = np.append(smw, cal_sindex.nominal_value)
    smwerr = np.append(smwerr, cal_sindex.std_dev)
    
    try :
        logrhkprime = ss.montecarlo_SMWtoRHK(cal_sindex, teff, bv, nsamples=1000, lc="ms", verbose=False, plot=False)
    except :
        logrhkprime = ufloat(np.nan, np.nan)
    rhk = np.append(rhk, logrhkprime.nominal_value)
    rhkerr = np.append(rhkerr, logrhkprime.std_dev)

    if options.verbose :
        print("Spectrum {}/{} -> BJD = {:.8f}  S-index = {:.4f}+/-{:.4f}  SMW={}  log(R'HK)={}".format(i+1, len(fluxes), template["times"][i], sidx, esidx, cal_sindex, logrhkprime))

# Save s-index time series to output file
if options.output != "":
    if options.instrumental :
        spectrallib.save_time_series(options.output, template["times"]-2400000., sindex, sindexerr, xlabel="rjd", ylabel="sindex", yerrlabel="sindexerr", write_header_rows=True)
    else :
        spectrallib.save_time_series(options.output, template["times"]-2400000., smw, smwerr, xlabel="rjd", ylabel="sindex", yerrlabel="sindexerr", write_header_rows=True)

if options.outputRHK != "" :
    spectrallib.save_time_series(options.outputRHK, template["times"]-2400000., rhk, rhkerr, xlabel="rjd", ylabel="rhk", yerrlabel="sig_rhk", write_header_rows=True)

# Plot time series
if options.plot :
    if options.instrumental :
        plt.errorbar(template["times"]-2400000., sindex, yerr=sindexerr, fmt='o', color='k')
        plt.hlines(template_sindex, template["times"][0]-2400000., template["times"][-1]-2400000., ls="-", lw=3, color="darkgreen", label=r"Template S-index = {:.4f}$\pm${:.4f}".format(template_sindex,template_sindexerr))
        plt.fill_between(x=template["times"]-2400000., y1=np.full_like(template["times"]-2400000.,template_sindex+template_sindexerr), y2=np.full_like(template["times"]-2400000.,template_sindex-template_sindexerr), color= "darkgreen",alpha= 0.3)
        plt.ylabel(r"S-index", fontsize=20)
    else :
        plt.errorbar(template["times"]-2400000., smw, yerr=smwerr, fmt='o', color='k')
        plt.hlines(template_sindex, template["times"][0]-2400000., template["times"][-1]-2400000., ls="-", lw=3, color="darkgreen", label=r"Template S-index = {:.4f}$\pm${:.4f}".format(cal_template_sindex.nominal_value,cal_template_sindex.std_dev))
        plt.fill_between(x=template["times"]-2400000., y1=np.full_like(template["times"]-2400000.,cal_template_sindex.nominal_value+cal_template_sindex.std_dev), y2=np.full_like(template["times"]-2400000.,cal_template_sindex.nominal_value-cal_template_sindex.std_dev), color= "darkgreen",alpha= 0.3)
        plt.ylabel(r"S$_{\rm MW}$", fontsize=20)

    plt.xlabel(r"BJD-2400000", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    plt.show()


