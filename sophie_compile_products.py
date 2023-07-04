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
from copy import deepcopy

parser = OptionParser()
parser.add_option("-o", "--output_file", dest="output_file", help="Output file name",type='string',default="")
parser.add_option("-d", "--output_dacerdb_file", dest="output_dacerdb_file", help="Output DACE rdb file name",type='string',default="")
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


outbigtable = hstack(bigtable)
#print(outbigtable)

if options.output_file != "" :
    outbigtable.write(options.output_file, format="ascii", overwrite=True)

if options.output_dacerdb_file != "" and os.path.exists(options.output_file) :

    #    dace_product = "{}.rdb".format(options.output_file.split(".")[0])
    dace_product = options.output_dacerdb_file

    results = ascii.read(options.output_file, data_start=1)

    dacetbl = Table()
    # results possible columns:
    #file object bjd snr exptime berv airmass vrad svrad biss bisserr fwhm fwhmerr sindex sindexerr H-alpha H-alphaerr NaI NaIerr CaI CaIerr
    # DACE mandatory columns
    # rjd    vrad    svrad
    dacetbl['rjd'] = results['bjd'] - 2400000
    dacetbl['vrad'] = results['vrad']
    dacetbl['svrad'] = results['svrad']

    # DACE acceptable columns
    #fwhm    sig_fwhm    contrast     sig_contrast    bis_span    sig_bis_span     s_mw  sig_s     ha    sig_ha    na    sig_na    ca    sig_ca
    #rhk    sig_rhk    sn_caii    prot_m08    sig_prot_m08    prot_n84    sig_prot_n84    berv ccf_noise    drift_noise    cal_therror    cal_thfile
    if 'fwhm' in results.colnames :
        dacetbl['fwhm'] = results['fwhm']
    if 'fwhmerr' in results.colnames :
        dacetbl['sig_fwhm'] = results['fwhmerr']

    if 'biss' in results.colnames :
        dacetbl['bis_span'] = results['biss']
    if 'bisserr' in results.colnames :
        dacetbl['sig_bis_span'] = results['bisserr']

    if 'berv' in results.colnames :
        dacetbl['berv'] = results['berv']
        
    if 'sindex' in results.colnames :
        dacetbl['s_mw'] = results['sindex']
    if 'sindexerr' in results.colnames :
        dacetbl['sig_s'] = results['sindexerr']

    if 'H-alpha' in results.colnames :
        dacetbl['ha'] = results['H-alpha']
    if 'H-alphaerr' in results.colnames :
        dacetbl['sig_ha'] = results['H-alphaerr']

    if 'NaI' in results.colnames :
        dacetbl['na'] = results['NaI']
    if 'NaIerr' in results.colnames :
        dacetbl['sig_na'] = results['NaIerr']

    if 'CaI' in results.colnames :
        dacetbl['ca'] = results['CaI']
    if 'CaIerr' in results.colnames :
        dacetbl['sig_ca'] = results['CaIerr']

    dacetbl.write(dace_product, format='ascii', overwrite=True)
