"""
    Created on Nov 16 2021
    
    Description: espectro_ir (infrared) library to handle IGRINS data
    
    @author: Eder Martioli <emartioli@lna.br>, <martioli@iap.fr>
    
    Laboratorio Nacional de Astrofisica, Brazil
    Institut d'Astrophysique de Paris, France
    
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import os,sys

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import signal
import scipy.interpolate as sint
from copy import deepcopy

def load_spectrum(filename) :

    spectrum = {}
    
    hdu = fits.open(filename)
    hdr = hdu[0].header
    
    spectrum["header"] = deepcopy(hdr)

    flux = hdu[0].data
    wl = np.full_like(flux,np.nan)
    wl0 = hdr["CRVAL1"] / 10.
    dwl = hdr["CDELT1"] / 10.

    for i in range(len(wl)) :
        wl[i] = wl0 + i * dwl

    spectrum["wl"] = wl
    spectrum["flux"] = flux
    spectrum["fluxerr"] = flux * 0.
    
    hdu.close()
    
    return spectrum


def write_spectrum_to_fits(waves, fluxes, fluxerrs, filename, header=None):
    """
        Description: function to save the spectrum to a fits file
        """
    
    if header is None :
        header = fits.Header()

    header.set('TTYPE1', "WAVE")
    header.set('TUNIT1', "NM")
    header.set('TTYPE2', "FLUX")
    header.set('TUNIT2', "COUNTS")
    header.set('TTYPE2', "FLUXERR")
    header.set('TUNIT2', "COUNTS")

    primary_hdu = fits.PrimaryHDU(header=header)
    hdu_wl = fits.ImageHDU(data=waves, name="WAVE")
    hdu_flux = fits.ImageHDU(data=fluxes, name="FLUX")
    hdu_err = fits.ImageHDU(data=fluxerrs, name="FLUXERR")
    mef_hdu = fits.HDUList([primary_hdu, hdu_wl, hdu_flux, hdu_err])

    mef_hdu.writeto(filename, overwrite=True)
