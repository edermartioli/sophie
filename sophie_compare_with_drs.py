# -*- coding: iso-8859-1 -*-
"""
    Created on Jan 10 2023
    
    Description: Script to compare results with the DRS
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python /Volumes/Samsung_T5/Science/sophie/sophie_compare_with_drs.py --input_drs_data='/Volumes/Samsung_T5/Science/TOI-1736/time-series/TOI-1736_spec_ts.txt' --input_rv_file='TOI-1736_sophie_ccfrv.rdb' --input_bis_file='TOI-1736_sophie_ccfbis.rdb' --input_fwhm_file='TOI-1736_sophie_ccffwhm.rdb'
    
    python /Volumes/Samsung_T5/Science/sophie/sophie_compare_with_drs.py --input_drs_data='/Volumes/Samsung_T5/Science/TOI-2141/time-series/TOI-2141_spec_ts.txt' --input_rv_file='TOI-2141_sophie_ccfrv.rdb' --input_bis_file='TOI-2141_sophie_ccfbis.rdb' --input_fwhm_file='TOI-2141_sophie_ccffwhm.rdb'

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
from scipy import optimize



def berv_model (coeffs, berv):
    outmodel = coeffs[1] * berv + coeffs[0]
    return outmodel
        
def berv_errfunc (coeffs, rvs, berv) :
    residuals = rvs - berv_model (coeffs, berv)
    return residuals



parser = OptionParser()
parser.add_option("-i", "--input_drs_data", dest="input_drs_data", help="Input drs data",type='string',default="")
parser.add_option("-t", "--input_rv_file", dest="input_rv_file", help="Input RV file (rdb format)",type='string',default="")
parser.add_option("-b", "--input_bis_file", dest="input_bis_file", help="Input bisector file (rdb format)",type='string',default="")
parser.add_option("-f", "--input_fwhm_file", dest="input_fwhm_file", help="Input FWHM file (rdb format)",type='string',default="")
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h sophie_compare_with_drs.py")
    sys.exit(1)

if options.verbose:
    print('Input drs data: ', options.input_drs_data)

timeseriesdata = ascii.read(options.input_drs_data)
files = timeseriesdata["dpr_type"]
time, fwhm, biss = timeseriesdata["bjd"], timeseriesdata["fwhm"], timeseriesdata["biss"]
rv, rverr = timeseriesdata["vrad"], timeseriesdata["svrad"]
berv = timeseriesdata["berv"]*1000

rvdata = ascii.read(options.input_rv_file, data_start=2)
bisdata = ascii.read(options.input_bis_file, data_start=2)
fwhmdata = ascii.read(options.input_fwhm_file, data_start=2)

mytime, myrv, myrverr = rvdata['rjd'], rvdata['vrad'], rvdata['svrad']
mybiss, mybisserr = bisdata['biss'], bisdata['bisserr']
myfwhm, myfwhmerr = fwhmdata['fwhm'], fwhmdata['fwhmerr']

"""
for i in range(len(time)) :
    matched = False
    for j in range(len(mytime)) :
        deltat = np.abs(time[i] - mytime[j])
        if deltat < 0.001 :
            #print("{} i={} match j={} time={:.10f} deltat={:.10f}".format(files[i], i, j, time[i], deltat))
            matched = True
    if not matched :
        print("File {} did not match!".format(files[i]))
"""

#guess = [0.0001, 1.001]
#pfit, success = optimize.leastsq(berv_errfunc, guess, args=(myrv, berv))
#plt.errorbar(time,myrv, yerr=myrverr, fmt='ko', label=r"RV$_{\rm DRS}$ - RV")
#plt.plot(time, berv_model(pfit, berv), 'r:', label="{:.4f}*BERV + {:.4f}".format(pfit[1],pfit[0]))
#plt.show()

rfdiff, rvdifferr = (rv - myrv)*1000, 1000*np.sqrt(rverr**2 + myrverr**2)

guess = [0.0001, 1.001]
pfit, success = optimize.leastsq(berv_errfunc, guess, args=(rfdiff, berv))


plt.errorbar(time,rfdiff, yerr=rvdifferr, fmt='ko', label=r"RV$_{\rm DRS}$ - RV")
#plt.plot(time, berv_model(pfit, berv), 'r:', label="{:.5f}*BERV + {:.5f}".format(pfit[1],pfit[0]))
plt.xlabel(r"BJD - 2400000",fontsize=20)
plt.ylabel(r"RV$_{\rm DRS}$ - RV [m/s]",fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=16)
plt.show()

plt.errorbar(time,1000*(biss - mybiss),yerr=1000*mybisserr,fmt='ko')
plt.xlabel(r"BJD - 2400000",fontsize=20)
plt.ylabel(r"Vs$_{\rm DRS}$ - Vs [m/s]",fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend()
plt.show()

fwhm_diff = 1000*(fwhm - myfwhm)
guess = [0.0001, 1.001]
fpfit, success = optimize.leastsq(berv_errfunc, guess, args=(fwhm_diff, berv))

plt.errorbar(time,fwhm_diff,yerr=1000*myfwhmerr,fmt='ko')
#plt.plot(time, berv_model(fpfit, berv), 'r:', label="{:.5f}*BERV + {:.5f}".format(fpfit[1],fpfit[0]))
plt.xlabel(r"BJD - 2400000",fontsize=20)
plt.ylabel(r"FWHM$_{\rm DRS}$ - FWHM [m/s]",fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend()
plt.show()
