# -*- coding: iso-8859-1 -*-
"""
    Created on Nov 18 2022
    
    Description: This routine runs several routines to anlyze the SOPHIE spectra
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python /Volumes/Samsung_T5/Science/sophie/sophie_pipeline.py --input=SOPHIE*s1d_A.fits -pv

    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys
import glob
import astropy.io.fits as fits

parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="Input spectral e.fits data pattern",type='string',default="*s1d_A.fits")
parser.add_option("-m", "--ccfmask", dest="ccfmask", help="Input CCF mask",type='string',default="")
parser.add_option("-n", "--object_name", dest="object_name", help="Object name",type='string',default="")
parser.add_option("-f", action="store_true", dest="force_reduction", help="force reduction", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h sophie_pipeline.py")
    sys.exit(1)

if options.verbose:
    print('SOPHIE spectral data pattern: ', options.input)
    print('Input object name: ', options.object_name)
    print('CCF mask: ', options.ccfmask)

sophie_dir = os.path.dirname(__file__) + '/'

if options.ccfmask == "" :
    options.ccfmask = os.path.join(sophie_dir,"masks/G2_nm.mas")

plot_flag = ""
if options.plot :
    plot_flag = "-p"
verbose_flag = ""
if options.verbose :
    verbose_flag = "-v"

object_name = options.object_name

if options.object_name == "" :
    inputdata = sorted(glob.glob(options.input))
    basehdu = fits.open(inputdata[0])
    object_name = "UNKNOWN"
    try :
        object_name = (basehdu[0].header["HIERARCH OHP TARG NAME"]).replace(" ","")
    except :
        print("WARNING: couldn't find object name in the header, continuing with name: {}".format(object_name))
    #for i in range(len(inputdata)) :
    #    hdu = fits.open(inputdata[i])

#############################
##### RUN CCF analysis ######
#############################
rv_file = "{}_sophie_ccf.rdb".format(object_name)
command = "python {0}sophie_ccf_pipeline.py --input={1} --ccf_mask={2} --output_rv_file={3} {4} {5}".format(sophie_dir, options.input, options.ccfmask, rv_file, plot_flag, verbose_flag)
print("Running: ",command)
if not os.path.exists(rv_file) or options.force_reduction :
    os.system(command)

#############################
##### RUN template ##########
#############################
template_file = "{}_sophie_template.fits".format(object_name)
command = "python {0}sophie_template.py --input={1} --rv_file={2} --output={3} {4} {5}".format(sophie_dir, options.input, rv_file, template_file, plot_flag, verbose_flag)
print("Running: ",command)
if not os.path.exists(template_file) or options.force_reduction :
    os.system(command)


#############################
######## RUN S-index ########
#############################
sindex_file = "{}_sophie_sindex.txt".format(object_name)
command = "python {0}sophie_sindex.py --input={1} --output={2} {3} {4}".format(sophie_dir, template_file, sindex_file, plot_flag, verbose_flag)
print("Running: ",command)
if not os.path.exists(sindex_file) or options.force_reduction :
    os.system(command)

