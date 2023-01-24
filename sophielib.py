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
import warnings

def sophie_order_limits() :
    order_limits = [388.600,393.617,398.046,402.655,
                    407.274,412.332,417.025,422.204,
                    427.587,433.052,438.122,443.875,
                    449.941,455.761,461.769,468.251,
                    474.594,481.484,487.854,494.826,
                    502.598,510.073,517.509,524.799,
                    533.323,540.436,549.862,559.032,
                    567.650,577.688,587.458,597.984,
                    608.643,619.317,630.540,642.296,
                    654.603,667.356,680.708]
    return order_limits

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


def get_wave(hdr, use_angstrom=False) :

    ordersize = hdr["NAXIS1"]
    norders = hdr["NAXIS2"]

    gorder = hdr["HIERARCH OHP DRS CAL TH GUESS ORDER"]

    x = np.arange(ordersize)
    
    ncoeffs = hdr['HIERARCH OHP DRS CAL TH DEG LL'] + 1

    order_wl = []
    
    for i in range(gorder+2) :
    
        coeffs = np.zeros_like(np.arange(ncoeffs),dtype=float)
        
        for j in range(ncoeffs) :
            if i == 0 :
                coeffs[ncoeffs-1-j] = hdr['HIERARCH OHP DRS CAL TH COEFF LL{:00d}'.format(j)]
            else :
                coeffs[ncoeffs-1-j] = hdr['HIERARCH OHP DRS CAL TH COEFF LL{:00d}{:00d}'.format(i,j)]
        #order = i*2
        ll = np.poly1d(coeffs)
        order_wl.append(ll(x))
        
        if i < gorder+1 :
            for j in range(ncoeffs) :
                if i == 0 :
                    coeffs[ncoeffs-1-j] = hdr['HIERARCH OHP DRS CAL TH COEFF LL{:00d}'.format(j+ncoeffs)]
                else :
                    coeffs[ncoeffs-1-j] = hdr['HIERARCH OHP DRS CAL TH COEFF LL{:00d}{:00d}'.format(i,j+ncoeffs)]
    
            ll = np.poly1d(coeffs)
            order_wl.append(ll(x))
            
    order_wl = np.array(order_wl)
    
    if use_angstrom :
        return order_wl
    else :
        return order_wl / 10

    
def load_e2ds_spectrum(filename) :

    spectrum = {}
    
    hdu = fits.open(filename)
    hdr = hdu[0].header
    
    spectrum["header"] = deepcopy(hdr)

    flux = hdu[0].data
    wl = get_wave(hdr)
 
    for order in range(len(wl)) :
        plt.plot(wl[order],flux[order])
    plt.show()
    
    exit()
 
    spectrum["wl"] = wl
    spectrum["flux"] = flux
    spectrum["fluxerr"] = np.sqrt(flux)
    
    hdu.close()
    
    return spectrum

def load_order_of_e2ds_spectrum(filename, order=0) :
    spectrum = {}
    
    hdu = fits.open(filename)
    hdr = hdu[0].header
    
    spectrum["header"] = deepcopy(hdr)

    flux = deepcopy(hdu[0].data[order])
    wl = get_wave(hdr)[order]
 
    spectrum["wl"] = wl
    spectrum["flux"] = flux
    spectrum["fluxerr"] = np.sqrt(flux)
    
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
