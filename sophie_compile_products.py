# -*- coding: iso-8859-1 -*-
"""
    Created on Jan 20 2023
    
    Description: Script to compile pipeline products into a single table
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python /Volumes/Samsung_T5/Science/sophie/sophie_compile_products.py --output_file=TOI-1736_results.txt --obslog_file=TOI-1736_sophie_obslog.rdb --rv_file=TOI-1736_sophie_ccfrv.rdb --bis_file=TOI-1736_sophie_ccfbis.rdb --fwhm_file=TOI-1736_sophie_ccffwhm.rdb

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

parser = OptionParser()
parser.add_option("-o", "--output_file", dest="output_file", help="Output file name",type='string',default="")
parser.add_option("-l", "--obslog_file", dest="obslog_file", help="Input observation log file",type='string',default="")
parser.add_option("-r", "--rv_file", dest="rv_file", help="Input RV file (rdb format)",type='string',default="")
parser.add_option("-b", "--bis_file", dest="bis_file", help="Input bisector file (rdb format)",type='string',default="")
parser.add_option("-f", "--fwhm_file", dest="fwhm_file", help="Input FWHM file (rdb format)",type='string',default="")

parser.add_option("-s", "--sindex_file", dest="sindex_file", help="Input S-index file (rdb format)",type='string',default="")
parser.add_option("-a", "--halpha_file", dest="halpha_file", help="Input H-alpha file (rdb format)",type='string',default="")
parser.add_option("-n", "--nai_file", dest="nai_file", help="Input Na I file (rdb format)",type='string',default="")
parser.add_option("-c", "--cai_file", dest="cai_file", help="Input Ca I file (rdb format)",type='string',default="")

parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h sophie_compile_products.py")
    sys.exit(1)


bigtable = []

if options.obslog_file != "" :
    if options.verbose:
        print('Adding observation log from file: ', options.obslog_file)
    obslog = ascii.read(options.obslog_file, data_start=2)
    bigtable.append(obslog)

if options.rv_file != "" :
    if options.verbose:
        print('Adding RVs from file: ', options.rv_file)
    rv = ascii.read(options.rv_file, data_start=2)
    del rv['rjd']
    bigtable.append(rv)

if options.bis_file != "" :
    if options.verbose:
        print('Adding Bisector span from file: ', options.bis_file)
    bis = ascii.read(options.bis_file, data_start=2)
    del bis['rjd']
    bigtable.append(bis)

if options.fwhm_file != "" :
    if options.verbose:
        print('Adding FWHM from file: ', options.fwhm_file)
    fwhm = ascii.read(options.fwhm_file, data_start=2)
    del fwhm['rjd']
    bigtable.append(fwhm)

if options.sindex_file != "" :
    if options.verbose:
        print('Adding S-index from file: ', options.sindex_file)
    sindex = ascii.read(options.sindex_file, data_start=2)
    del sindex['rjd']
    bigtable.append(sindex)

if options.halpha_file != "" :
    if options.verbose:
        print('Adding H-alpha from file: ', options.halpha_file)
    halpha = ascii.read(options.halpha_file, data_start=2)
    del halpha['rjd']
    bigtable.append(halpha)

if options.nai_file != "" :
    if options.verbose:
        print('Adding Na I from file: ', options.nai_file)
    nai = ascii.read(options.nai_file, data_start=2)
    del nai['rjd']
    bigtable.append(nai)

if options.cai_file != "" :
    if options.verbose:
        print('Adding Ca I from file: ', options.cai_file)
    cai = ascii.read(options.cai_file, data_start=2)
    del cai['rjd']
    bigtable.append(cai)

if options.output_file != "" :
    outbigtable = hstack(bigtable)
    print(outbigtable)
    outbigtable.write(options.output_file, format='ascii', overwrite=True)
    #ascii.write(outbigtable, options.output_file, format='csv', overwrite=True)
    #ascii.write(outbigtable, options.output_file, format='latex', overwrite=True)

