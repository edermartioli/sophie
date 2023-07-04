# -*- coding: utf-8 -*-
"""
# CODE NAME HERE
# CODE DESCRIPTION HERE
Created on 2020-03-27 at 13:42
@author: cook (adapted by E. Martioli on 2020-04-28)
"""
from astropy.io import fits
from astropy.table import vstack,Table
from astropy import units as uu
import numpy as np
import warnings
import sys
import os
from scipy.optimize import curve_fit
from scipy.stats import chisquare
import matplotlib.pyplot as plt
from astropy.io.fits.verify import VerifyWarning
import warnings
from scipy import constants
import glob

from scipy.interpolate import InterpolatedUnivariateSpline as ius

import reduc_lib
from copy import deepcopy

NORDERS = 39

helpstr = """
----------------------------------------------------------------------------
new_ccf_code.py
----------------------------------------------------------------------------
This code takes no arguments - you must edit the "variables section"  of the
code
1. Finding the correct calibrations
    a) For your observation find the date
    b) Go to your /calibDB/ directory
    c) find the correct file (closest in time?):
        BLAZE:  *blaze_{fiber}.fits
        WAVE:   *wave_night_{fiber}.fits
2. Finding masks

    a) go to your apero-drs installation directory
    b) go to data/spirou/ccf sub-directory
    c) use one of these masks
3. Two options for where you put files
    a) copy all data to a directory of your choice ({path})
        i) copy this code there and set W1='' and W2=''
        ii) or set W1={path} and W2={path}
        
    b) set the paths for your data (W1) and your mask directory (W2)
    
    Then update the filenames:
        IN_FILE: the e2dsff_C or e2dsff_tcorr _AB file
        BLAZE_FILE: the blaze file from calibDB - get the fiber correct!
        WAVE_FILE: the wave file from calibDB - get the fiber correct!
        MASK_FILE: the mask file
        
    Note there are two cases (set CASE=1 or CASE=2)
    
    For case=1 we assume your IN_FILE is a OBJ
    For case=2 we assume your IN_FILe is a FP
----------------------------------------------------------------------------
"""


# =============================================================================
# Define variables
# =============================================================================
# constants
SPEED_OF_LIGHT = 299792.458    # [km/s]

# whether to plot (True or False)
PLOT = True

def set_ccf_params(maskfile, telluric_masks=[], science_channel=True) :
    
    loc = {}
    loc["MASK_FILE"] = maskfile
    loc["TELL_MASK_FILES"] = []
    if len(telluric_masks) :
        for m in telluric_masks :
            loc["TELL_MASK_FILES"].append(m)
    loc["SOURCE_RV"] = 0.0
    # CCF is a set of Dirac functions
    #loc["KERNEL"] = None
    # boxcar length expressed in km/s
    # loc["KERNEL"] = ['boxcar', 5]
    # gaussian with e-width in km/s
    # loc["KERNEL"] = ['gaussian', 3.5]
    # supergaussian e-width + exponent
    #loc["KERNEL"] = ['supergaussian', 3.5, 4]

    if science_channel :
        loc["MASK_COLS"] = ['ll_mask_s', 'll_mask_e', 'w_mask']
        if maskfile.endswith(".nmas") :
            loc["MASK_COLS"] = ['order_mask', 'll_mask_s', 'll_mask_e', 'w_mask']
        
        # variables
        # These values are taken from the constants file
        loc["MASK_WIDTH"] = 0.5                   # CCF_MASK_WIDTH
        loc["MASK_MIN_WEIGHT"] = 0.0              # CCF_MASK_MIN_WEIGHT
        loc["CCF_STEP"] = 0.2                     # CCF_DEFAULT_STEP (or user input)
        #loc["CCF_WIDTH"] = 300                    # CCF_DEFAULT_WIDTH (or user input)
        loc["CCF_WIDTH"] = 100                     # CCF_DEFAULT_WIDTH (or user input)
        #loc["CCF_WIDTH"] = 60                     # CCF_DEFAULT_WIDTH (or user input)
        loc["CCF_RV_NULL"] = -9999.99             # CCF_OBJRV_NULL_VAL
        loc["CCF_N_ORD_MAX"] = NORDERS                 # CCF_N_ORD_MAX
        loc["BLAZE_NORM_PERCENTILE"] = 90         # CCF_BLAZE_NORM_PERCENTILE
        loc["BLAZE_THRESHOLD"] = 0.3              # WAVE_FP_BLAZE_THRES
        loc["IMAGE_PIXEL_SIZE"] = 2.28            # IMAGE_PIXEL_SIZE
        loc["NOISE_SIGDET"] = 8.0                 # CCF_NOISE_SIGDET
        loc["NOISE_SIZE"] = 12                    # CCF_NOISE_BOXSIZE
        loc["NOISE_THRES"] = 1.0e9                # CCF_NOISE_THRES
        loc["KERNEL"] = None
    else :
        # build file paths
        loc["MASK_COLS"] = ['ll_mask_s', 'll_mask_e', 'w_mask']
        # variables
        # These values are taken from the constants file
        loc["MASK_WIDTH"] = 1.7                   # CCF_MASK_WIDTH
        loc["MASK_MIN_WEIGHT"] = 0.0              # CCF_MASK_MIN_WEIGHT
        loc["CCF_STEP"] = 0.25                     # WAVE_CCF_STEP
        loc["CCF_WIDTH"] = 6.5                    # WAVE_CCF_WIDTH
        loc["CCF_RV_NULL"] = -9999.99             # CCF_OBJRV_NULL_VAL
        loc["CCF_N_ORD_MAX"] = NORDERS                 # WAVE_CCF_N_ORD_MAX
        loc["BLAZE_NORM_PERCENTILE"] = 90         # CCF_BLAZE_NORM_PERCENTILE
        loc["BLAZE_THRESHOLD"] = 0.3              # WAVE_FP_BLAZE_THRES
        loc["IMAGE_PIXEL_SIZE"] = 2.28            # IMAGE_PIXEL_SIZE
        loc["NOISE_SIGDET"] = 8.0                 # WAVE_CCF_NOISE_SIGDET
        loc["NOISE_SIZE"] = 12                    # WAVE_CCF_NOISE_BOXSIZE
        loc["NOISE_THRES"] = 1.0e9                # WAVE_CCF_NOISE_THRES
        loc["KERNEL"] = ['gaussian', 1.4]

    return loc

# =============================================================================
# Define functions
# =============================================================================
def read_mask(mask_file, mask_cols):
    table = Table.read(mask_file, format='ascii')
    # get column names
    oldcols = list(table.colnames)
    # rename columns
    for c_it, col in enumerate(mask_cols):
        table[oldcols[c_it]].name = col
    # return table
    return table


def get_mask(table, mask_width, mask_min, mask_units='nm'):
    ll_mask_e = np.array(table['ll_mask_e']).astype(float)
    ll_mask_s = np.array(table['ll_mask_s']).astype(float)
    ll_mask_d = ll_mask_e - ll_mask_s
    ll_mask_ctr = ll_mask_s + ll_mask_d * 0.5
    if "order_mask" in table :
        order_mask = np.array(table['order_mask']).astype(float)
    # if mask_width > 0 ll_mask_d is multiplied by mask_width/c
    if mask_width > 0:
        ll_mask_d = mask_width * ll_mask_s / SPEED_OF_LIGHT
    # make w_mask an array
    w_mask = np.array(table['w_mask']).astype(float)
    # use w_min to select on w_mask or keep all if w_mask_min >= 1
    if mask_min < 1.0:
        mask = w_mask > mask_min
        ll_mask_d = ll_mask_d[mask]
        ll_mask_ctr = ll_mask_ctr[mask]
        w_mask = w_mask[mask]
        if "order_mask" in table :
            order_mask = order_mask[mask]

    # else set all w_mask to one (and use all lines in file)
    else:
        w_mask = np.ones(len(ll_mask_d))
    # ----------------------------------------------------------------------
    # deal with the units of ll_mask_d and ll_mask_ctr
    # must be returned in nanometers
    # ----------------------------------------------------------------------
    # get unit object from mask units string
    unit = getattr(uu, mask_units)
    # add units
    ll_mask_d = ll_mask_d * unit
    ll_mask_ctr = ll_mask_ctr * unit
    # convert to nanometers
    ll_mask_d = ll_mask_d.to(uu.nm).value
    ll_mask_ctr = ll_mask_ctr.to(uu.nm).value
    # ----------------------------------------------------------------------
    # return the size of each pixel, the central point of each pixel
    #    and the weight mask
    if "order_mask" in table :
        return order_mask, ll_mask_d, ll_mask_ctr, w_mask
    else :
        return ll_mask_d, ll_mask_ctr, w_mask


def relativistic_waveshift(dv, units='km/s'):
    """
    Relativistic offset in wavelength
    default is dv in km/s
    :param dv: float or numpy array, the dv values
    :param units: string or astropy units, the units of dv
    :return:
    """
    # get c in correct units
    # noinspection PyUnresolvedReferences
    if units == 'km/s' or units == uu.km/uu.s:
        c = SPEED_OF_LIGHT
    # noinspection PyUnresolvedReferences
    elif units == 'm/s' or units == uu.m/uu.s:
        c = SPEED_OF_LIGHT * 1000
    else:
        raise ValueError("Wrong units for dv ({0})".format(units))
    # work out correction
    corrv = np.sqrt((1 + dv / c) / (1 - dv / c))
    # return correction
    return corrv


def iuv_spline(x, y, **kwargs):
    # check whether weights are set
    w = kwargs.get('w', None)
    # copy x and y
    x, y = np.array(x), np.array(y)
    # find all NaN values
    nanmask = ~np.isfinite(y)

    if np.sum(~nanmask) < 2:
        y = np.zeros_like(x)
    elif np.sum(nanmask) == 0:
        pass
    else:
        # replace all NaN's with linear interpolation
        badspline = ius(x[~nanmask], y[~nanmask],
                                                 k=1, ext=1)
        y[nanmask] = badspline(x[nanmask])
    # return spline
    return ius(x, y, **kwargs)


def fit_ccf(rv, ccf, fit_type, verbose=False):
    """
    Fit the CCF to a guassian function
    :param rv: numpy array (1D), the radial velocities for the line
    :param ccf: numpy array (1D), the CCF values for the line
    :param fit_type: int, if "0" then we have an absorption line
                          if "1" then we have an emission line
    :return result: numpy array (1D), the fit parameters in the
                    following order:
                [amplitude, center, fwhm, offset from 0 (in y-direction)]
    :return ccf_fit: numpy array (1D), the fit values, i.e. the gaussian values
                     for the fit parameters in "result"
    """
    # deal with inconsistent lengths
    if len(rv) != len(ccf):
        print('\tERROR: RV AND CCF SHAPE DO NOT MATCH')
        sys.exit()

    # deal with all nans
    if np.sum(np.isnan(ccf)) == len(ccf):
        # log warning about all NaN ccf
        if verbose :
            print('\tWARNING: NANS in CCF')
        # return NaNs
        result = np.zeros(4) * np.nan
        ccf_fit = np.zeros_like(ccf) * np.nan
        return result, ccf_fit

    # get constants
    max_ccf, min_ccf = np.nanmax(ccf), np.nanmin(ccf)
    argmin, argmax = np.nanargmin(ccf), np.nanargmax(ccf)
    diff = max_ccf - min_ccf
    rvdiff = rv[1] - rv[0]
    # set up guess for gaussian fit
    # if fit_type == 0 then we have absorption lines
    if fit_type == 0:
        if np.nanmax(ccf) != 0:
            a = np.array([-diff / max_ccf, rv[argmin], 4 * rvdiff, 0])
        else:
            a = np.zeros(4)
    # else (fit_type == 1) then we have emission lines
    else:
        a = np.array([diff / max_ccf, rv[argmax], 4 * rvdiff, 1])
    # normalise y
    y = ccf / max_ccf - 1 + fit_type
    # x is just the RVs
    x = rv
    # uniform weights
    w = np.ones(len(ccf))
    # get gaussian fit
    nanmask = np.isfinite(y)
    y[~nanmask] = 0.0
    # fit the gaussian
    try:
        with warnings.catch_warnings(record=True) as _:
            result, fit = fitgaussian(x, y, weights=w, guess=a)
    except RuntimeError:
        result = np.repeat(np.nan, 4)
        fit = np.repeat(np.nan, len(x))

    # scale the ccf
    ccf_fit = (fit + 1 - fit_type) * max_ccf

    # return the best guess and the gaussian fit
    return result, ccf_fit


def gauss_function(x, a, x0, sigma, dc):
    """
    A standard 1D gaussian function (for fitting against)]=
    :param x: numpy array (1D), the x data points
    :param a: float, the amplitude
    :param x0: float, the mean of the gaussian
    :param sigma: float, the standard deviation (FWHM) of the gaussian
    :param dc: float, the constant level below the gaussian
    :return gauss: numpy array (1D), size = len(x), the output gaussian
    """
    return a * np.exp(-0.5 * ((x - x0) / sigma) ** 2) + dc


def fitgaussian(x, y, weights=None, guess=None, return_fit=True,
                return_uncertainties=False):
    """
    Fit a single gaussian to the data "y" at positions "x", points can be
    weighted by "weights" and an initial guess for the gaussian parameters
    :param x: numpy array (1D), the x values for the gaussian
    :param y: numpy array (1D), the y values for the gaussian
    :param weights: numpy array (1D), the weights for each y value
    :param guess: list of floats, the initial guess for the guassian fit
                  parameters in the following order:
                  [amplitude, center, fwhm, offset from 0 (in y-direction)]
    :param return_fit: bool, if True also calculates the fit values for x
                       i.e. yfit = gauss_function(x, *pfit)
    :param return_uncertainties: bool, if True also calculates the uncertainties
                                 based on the covariance matrix (pcov)
                                 uncertainties = np.sqrt(np.diag(pcov))
    :return pfit: numpy array (1D), the fit parameters in the
                  following order:
                [amplitude, center, fwhm, offset from 0 (in y-direction)]
    :return yfit: numpy array (1D), the fit y values, i.e. the gaussian values
                  for the fit parameters, only returned if return_fit = True
    """

    # if we don't have weights set them to be all equally weighted
    if weights is None:
        weights = np.ones(len(x))
    weights = 1.0 / weights
    # if we aren't provided a guess, make one
    if guess is None:
        guess = [np.nanmax(y), np.nanmean(y), np.nanstd(y), 0]
    # calculate the fit using curve_fit to the function "gauss_function"
    with warnings.catch_warnings(record=True) as _:
        pfit, pcov = curve_fit(gauss_function, x, y, p0=guess, sigma=weights,
                               absolute_sigma=True)
    if return_fit and return_uncertainties:
        # calculate the fit parameters
        yfit = gauss_function(x, *pfit)
        # work out the normalisation constant
        chis, _ = chisquare(y, f_exp=yfit)
        norm = chis / (len(y) - len(guess))
        # calculate the fit uncertainties based on pcov
        efit = np.sqrt(np.diag(pcov)) * np.sqrt(norm)
        # return pfit, yfit and efit
        return pfit, yfit, efit
    # if just return fit
    elif return_fit:
        # calculate the fit parameters
        yfit = gauss_function(x, *pfit)
        # return pfit and yfit
        return pfit, yfit
    # if return uncertainties
    elif return_uncertainties:
        # calculate the fit parameters
        yfit = gauss_function(x, *pfit)
        # work out the normalisation constant
        chis, _ = chisquare(y, f_exp=yfit)
        norm = chis / (len(y) - len(guess))
        # calculate the fit uncertainties based on pcov
        efit = np.sqrt(np.diag(pcov)) * np.sqrt(norm)
        # return pfit and efit
        return pfit, efit
    # else just return the pfit
    else:
        # return pfit
        return pfit


def delta_v_rms_2d(spe, wave, sigdet, threshold, size):
    """
    Compute the photon noise uncertainty for all orders (for the 2D image)
    :param spe: numpy array (2D), the extracted spectrum
                size = (number of orders by number of columns (x-axis))
    :param wave: numpy array (2D), the wave solution for each pixel
    :param sigdet: float, the read noise (sigdet) for calculating the
                   noise array
    :param threshold: float, upper limit for pixel values, above this limit
                      pixels are regarded as saturated
    :param size: int, size (in pixels) around saturated pixels to also regard
                 as bad pixels
    :return dvrms2: numpy array (1D), the photon noise for each pixel (squared)
    :return weightedmean: float, weighted mean photon noise across all orders
    """
    # flag (saturated) fluxes above threshold as "bad pixels"
    with warnings.catch_warnings(record=True) as _:
        flag = spe < threshold
    # flag all fluxes around "bad pixels" (inside +/- size of the bad pixel)
    for i_it in range(1, 2 * size, 1):
        flag[size:-size] *= flag[i_it: i_it - 2 * size]
    # get the wavelength normalised to the wavelength spacing
    nwave = wave[1:-1] / (wave[2:] - wave[:-2])
    # get the flux + noise array
    sxn = (spe[1:-1] + sigdet ** 2)
    # get the flux difference normalised to the flux + noise
    nspe = (spe[2:] - spe[:-2]) / sxn
    # get the mask value
    maskv = flag[2:] * flag[1:-1] * flag[:-2]
    # get the total
    tot = np.nansum(sxn * ((nwave * nspe) ** 2) * maskv)
    # convert to dvrms2
    with warnings.catch_warnings(record=True) as _:
        dvrms2 = ((SPEED_OF_LIGHT * 1000) ** 2) / abs(tot)
    # weighted mean of dvrms2 values
    weightedmean = 1. / np.sqrt(np.nansum(1.0 / dvrms2))
    # return dv rms and weighted mean
    return dvrms2, weightedmean


def fwhm(sigma=1.0):
    """
    Get the Full-width-half-maximum value from the sigma value (~2.3548)
    :param sigma: float, the sigma, default value is 1.0 (normalised gaussian)
    :return: 2 * sqrt(2 * log(2)) * sigma = 2.3548200450309493 * sigma
    """
    return 2 * np.sqrt(2 * np.log(2)) * sigma


def ccf_calculation(p, wave, flux, blaze, targetrv, mask_centers, mask_weights,
                    berv, fit_type, verbose=False):
    # get rvmin and rvmax
    rvmin = targetrv - p['CCF_WIDTH']
    rvmax = targetrv + p['CCF_WIDTH'] + p['CCF_STEP']
    # get the dimensions
    nbpix = len(flux)
    # create a rv ccf range
    rv_ccf = np.arange(rvmin, rvmax, p['CCF_STEP'])
    
    kernel = p['KERNEL']
    # if we have defined 'kernel', it must be a list
    # with the first element being the type of convolution
    # and subsequent arguments being parameters. For now,
    # we have :
    #
    #  --> boxcar convolution
    # ['boxcar', width]
    #
    # kernel = [1, 1, ...., 1, 1]
    #
    # --> gaussian convolution
    #
    # ['gaussian', e-width]
    # kernel = exp( -0.5*(x/ew)**2 )
    #
    # --> super gaussian
    #
    # ['supergaussian', e-width, beta]
    #
    # kernel = exp( -0.5*(x/ew)**beta )
    #
    # Other functions could be added below
    #
    if isinstance(kernel, list):
        if kernel[0] == 'boxcar':
            # ones with a length of kernel[1]
            ker = np.ones(int(np.round(kernel[1] / p['CCF_STEP'])))
        elif kernel[0] == 'gaussian':
            # width of the gaussian expressed in
            # steps of CCF
            ew = kernel[1] / p['CCF_STEP']
            index = np.arange(-4 * np.ceil(ew), 4 * np.ceil(ew) + 1)
            ker = np.exp(-0.5 * (index / ew) ** 2)
        elif kernel[0] == 'supergaussian':
            # width of the gaussian expressed in
            # steps of CCF. Exponents should be
            # between 0.1 and 10.. Values above
            # 10 are (nearly) the same as a boxcar.
            if (kernel[1] < 0.1) or (kernel[1] > 10):
                raise ValueError('CCF ERROR: kernel[1] is out of range.')

            ew = kernel[1] / p['CCF_STEP']

            index = np.arange(-4 * np.ceil(ew), 4 * np.ceil(ew) + 1)
            ker = np.exp(-0.5 * np.abs(index / ew) ** kernel[2])

        else:
                # kernel name is not known - generate error
            raise ValueError('CCF ERROR: name of kernel not accepted!')

        ker = ker / np.sum(ker)

        if len(ker) > (len(rv_ccf)-1):
            # TODO : give a proper error
            err_msg = """
            The size of your convolution kernel is too big for your
            CCF size. Please either increase the CCF_WIDTH value or
            decrease the width of your convolution kernel. In boxcar,
            this implies a length bigger than CCF_WIDTH/CCF_STEP, in
            gaussian and supergaussian, this means that
            CCF_WIDTH/CCF_STEP is >8*ew. The kernel has to run from
            -4 sigma to +4 sigma.
            """
            raise ValueError('CCF ERROR: {0}'.format(err_msg))
    
    # ----------------------------------------------------------------------

    # COMMENT EA, normalization moved before the masking
    #
    # normalize per-ord blaze to its peak value
    # this gets rid of the calibration lamp SED
    blaze /= np.nanpercentile(blaze, p['BLAZE_NORM_PERCENTILE'])
    # COMMENT EA, changing NaNs to 0 in the blaze
    blaze[np.isfinite(blaze) == 0] = 0
    # mask on the blaze
    with warnings.catch_warnings(record=True) as _:
        blazemask = blaze > p['BLAZE_THRESHOLD']
    # get order mask centers and mask weights
    min_wav = np.nanmin(wave[blazemask])
    max_wav = np.nanmax(wave[blazemask])
    # COMMENT EA there's a problem with the sign in the min/max
    min_wav = min_wav * (1 - rvmin / SPEED_OF_LIGHT)
    max_wav = max_wav * (1 - rvmax / SPEED_OF_LIGHT)
    # mask the ccf mask by the order length
    mask_wave_mask = (mask_centers > min_wav)
    mask_wave_mask &= (mask_centers < max_wav)

    # ------------------------------------------------------------------
    # find any places in spectrum or blaze where pixel is NaN
    nanmask = np.isnan(flux) | np.isnan(blaze)
    # ------------------------------------------------------------------
    # deal with no valid lines
    if np.sum(mask_wave_mask) == 0:
        if verbose:
            print('\tWARNING: MASK INVALID FOR WAVELENGTH RANGE --> NAN')

    # ------------------------------------------------------------------
    # deal with orders where number of nans is too large (> 90% of order size)
    if np.sum(nanmask) > nbpix * 0.9:
        if verbose:
            print('\tWARNING: ALL SP OR BLZ NAN --> NAN')

    # ------------------------------------------------------------------
    # set the spectrum or blaze NaN pixels to zero (dealt with by divide)
    flux[nanmask] = 0
    blaze[nanmask] = 0
    # now every value that is zero is masked (we don't want to spline these)
    good = (flux != 0) & (blaze != 0)
    # ------------------------------------------------------------------
    # spline the spectrum and the blaze
    spline_sp = iuv_spline(wave[good], flux[good], k=5, ext=1)
    spline_bl = iuv_spline(wave[good], blaze[good], k=5, ext=1)
    # ------------------------------------------------------------------
    # set up the ccf for this order
    ccf = np.zeros_like(rv_ccf)
    # ------------------------------------------------------------------
    # get the wavelength shift (dv) in relativistic way
    wave_shifts = relativistic_waveshift(rv_ccf - berv)
    # ------------------------------------------------------------------
    # set number of valid lines used to zero
    numlines = 0
    # loop around the rvs and calculate the CCF at this point
    part3 = spline_bl(mask_centers)
    for rv_element in range(len(rv_ccf)):
        wave_tmp = mask_centers * wave_shifts[rv_element]
        part1 = spline_sp(wave_tmp)
        part2 = spline_bl(wave_tmp)
        numlines = np.sum(spline_bl(wave_tmp) != 0)
        # CCF is the division of the sums
        with warnings.catch_warnings(record=True) as _:
            ccf_element = ((part1 * part3) / part2) * mask_weights
            ccf[rv_element] = np.nansum(ccf_element)
    # ------------------------------------------------------------------
    # deal with NaNs in ccf
    if np.sum(np.isnan(ccf)) > 0:
        # log all NaN
        if verbose:
            print('WARNING: CCF is NAN')

    # ------------------------------------------------------------------
    # Convolve by the appropriate CCF kernel, if any
    if type(kernel) == list:
        weight = np.convolve(np.ones(len(ccf)), ker, mode='same')
        ccf = np.convolve(ccf, ker, mode='same') / weight
    # ------------------------------------------------------------------

    # normalise CCF to median
    ccf_norm = ccf / np.nanmedian(ccf)

    # ------------------------------------------------------------------
    # fit the CCF with a gaussian
    fargs = [rv_ccf, ccf, fit_type]
    ccf_coeffs, ccf_fit = fit_ccf(*fargs)
    # ------------------------------------------------------------------
    # calculate the residuals of the ccf fit
    res = ccf - ccf_fit
    # calculate the CCF noise per order
    ccf_noise = np.array(res)
    # calculate the snr for this order
    ccf_snr = np.abs(ccf_coeffs[0] / np.nanmedian(np.abs(ccf_noise)))
    # ------------------------------------------------------------------
    
    # store outputs in param dict
    props = dict()
    props['RV_CCF'] = rv_ccf
    props['CCF'] = ccf
    props['CCF_LINES'] = numlines
    props['TOT_LINE'] = numlines
    props['CCF_NOISE'] = ccf_noise
    props['CCF_SNR'] = ccf_snr
    props['CCF_FIT'] = ccf_fit
    props['CCF_FIT_COEFFS'] = ccf_coeffs
    props['CCF_NORM'] = ccf_norm

    # Return properties
    return props


def mean_ccf(p, props, targetrv, fit_type, normalize_ccfs=True, plot=False, verbose=False):

    m_ccf = props['CCF_NORM']

    # get the fit for the normalized average ccf
    mean_ccf_coeffs, mean_ccf_fit = fit_ccf(props['RV_CCF'],
                                            m_ccf, fit_type=fit_type)

    if plot :
        plt.plot(props['RV_CCF'], m_ccf / np.nanmedian(m_ccf), 'r--', linewidth=2, label="Mean CCF")
        plt.plot(props['RV_CCF'], mean_ccf_fit / np.nanmedian(mean_ccf_fit), 'g-', linewidth=2, label="Model CCF")

        plt.xlabel(r"Velocity [km/s]")
        plt.ylabel(r"Relative flux")
        plt.legend()
        plt.show()

    # get the RV value from the normalised average ccf fit center location
    ccf_rv = float(mean_ccf_coeffs[1])
    # get the contrast (ccf fit amplitude)
    ccf_contrast = np.abs(100 * mean_ccf_coeffs[0])
    # get the FWHM value
    ccf_fwhm = mean_ccf_coeffs[2] * fwhm()
    # --------------------------------------------------------------------------
    #  CCF_NOISE uncertainty
    ccf_noise_tot = np.sqrt(np.nanmean(props['CCF_NOISE'] ** 2))
    # Calculate the slope of the CCF
    average_ccf_diff = (m_ccf[2:] - m_ccf[:-2])
    rv_ccf_diff = (props['RV_CCF'][2:] - props['RV_CCF'][:-2])
    ccf_slope = average_ccf_diff / rv_ccf_diff
    # Calculate the CCF oversampling
    ccf_oversamp = p['IMAGE_PIXEL_SIZE'] / p['CCF_STEP']
    # create a list of indices based on the oversample grid size
    flist = np.arange(np.round(len(ccf_slope) / ccf_oversamp))
    indexlist = np.array(flist * ccf_oversamp, dtype=int)
    # we only want the unique pixels (not oversampled)
    indexlist = np.unique(indexlist)
    # get the rv noise from the sum of pixels for those points that are
    #     not oversampled
    keep_ccf_slope = ccf_slope[indexlist]
    rv_noise = np.nansum(keep_ccf_slope ** 2 / ccf_noise_tot ** 2) ** (-0.5)
    # --------------------------------------------------------------------------
    # log the stats
    wargs = [ccf_contrast, float(mean_ccf_coeffs[1]), rv_noise, ccf_fwhm]
    if verbose:
        print('MEAN CCF:')
        print('\tCorrelation: C={0:1f}[%] RV={1:.5f}[km/s] RV_NOISE={2:.5f}[km/s] '
          'FWHM={3:.4f}[km/s]'.format(*wargs))
    # --------------------------------------------------------------------------
    # add to output array
    props['MEAN_CCF'] = m_ccf
    props['MEAN_CCF_RES'] = mean_ccf_coeffs
    props['MEAN_CCF_FIT'] = mean_ccf_fit

    if np.isfinite(ccf_rv) :
        props['MEAN_RV'] = ccf_rv
        props['MEAN_CONTRAST'] = ccf_contrast
        props['MEAN_FWHM'] = ccf_fwhm
    else :
        props['MEAN_RV'] = -9999
        props['MEAN_CONTRAST'] = -9999
        props['MEAN_FWHM'] = -9999

    if np.isfinite(rv_noise) :
        props['MEAN_RV_NOISE'] = rv_noise
    else :
        props['MEAN_RV_NOISE'] = 0.
    # --------------------------------------------------------------------------
    # add constants to props
    props['CCF_MASK'] = os.path.basename(p['MASK_FILE'])
    props['CCF_STEP'] = p['CCF_STEP']
    props['CCF_WIDTH'] = p['CCF_WIDTH']
    props['TARGET_RV'] = targetrv
    props['CCF_SIGDET'] = p['NOISE_SIGDET']
    props['CCF_BOXSIZE'] = p['NOISE_SIZE']
    props['CCF_MAXFLUX'] = p['NOISE_THRES']
    props['CCF_NMAX'] = p['CCF_N_ORD_MAX']
    props['MASK_MIN'] = p['MASK_MIN_WEIGHT']
    props['MASK_WIDTH'] = p['MASK_WIDTH']
    props['MASK_UNITS'] = 'nm'
    # --------------------------------------------------------------------------
    return props



def write_file(props, infile, maskname, header, wheader, rv_drifts, output="", verbose=False):

    warnings.simplefilter('ignore', category=VerifyWarning)
    
    # produce CCF table
    table1 = Table()
    table1['RV'] = props['RV_CCF']
    for order_num in range(1):
        table1['ORDER{0:02d}'.format(order_num)] = props['CCF']
    table1['COMBINED'] = props['MEAN_CCF']
    # ----------------------------------------------------------------------
    # produce stats table
    table2 = Table()
    table2['ORDERS'] = np.arange(1).astype(int)
    table2['NLINES'] = [props['CCF_LINES']]
    # get the coefficients
    coeffs = props['CCF_FIT_COEFFS']
    table2['CONTRAST'] = [np.abs(100 * coeffs[0])]
    table2['RV'] = [coeffs[1]]
    table2['FWHM'] = [coeffs[2]]
    table2['DC'] = [coeffs[3]]
    table2['SNR'] = [props['CCF_SNR']]
    table2['NORM'] = [props['CCF_NORM']]

    header['SNR'] = (header['SNR'], 'Signal-to-noise ratio')
    header['BJD'] = (header['BJD'], 'Barycentric Julian Date')
   
    # ----------------------------------------------------------------------
    # add to the header
    # ----------------------------------------------------------------------
    # add results from the CCF
    header['CCFMNRV'] = (props['MEAN_RV'],
                         'Mean RV calc. from the mean CCF [km/s]')
    header['CCFMCONT'] = (props['MEAN_CONTRAST'],
                          'Mean contrast (depth of fit) from mean CCF')
    header['CCFMFWHM'] = (props['MEAN_FWHM'],
                          'Mean FWHM from mean CCF')
    header['CCFMRVNS'] = (props['MEAN_RV_NOISE'],
                          'Mean RV Noise from mean CCF')
    header['CCFTLINE'] = (props['TOT_LINE'],
                          'Total no. of mask lines used in CCF')
    # ----------------------------------------------------------------------
    # add constants used to process
    header['CCFMASK'] = (props['CCF_MASK'], 'CCF mask file used')
    header['CCFSTEP'] = (props['CCF_STEP'], 'CCF step used [km/s]')
    header['CCFWIDTH'] = (props['CCF_WIDTH'], 'CCF width used [km/s]')
    header['CCFTRGRV'] = (props['TARGET_RV'],
                          'CCF central RV used in CCF [km/s]')
    header['CCFSIGDT'] = (props['CCF_SIGDET'],
                          'Read noise used in photon noise calc. in CCF')
    header['CCFBOXSZ'] = (props['CCF_BOXSIZE'],
                          'Size of bad px used in photon noise calc. in CCF')
    header['CCFMAXFX'] = (props['CCF_MAXFLUX'],
                          'Flux thres for bad px in photon noise calc. in CCF')
    header['CCFORDMX'] = (props['CCF_NMAX'],
                          'Last order used in mean for mean CCF')
    header['CCFMSKMN'] = (props['MASK_MIN'],
                          'Minimum weight of lines used in the CCF mask')
    header['CCFMSKWD'] = (props['MASK_WIDTH'],
                          'Width of lines used in the CCF mask')
    header['CCFMUNIT'] = (props['MASK_UNITS'], 'Units used in CCF Mask')
    # ----------------------------------------------------------------------
    #header['RV_WAVFN'] = (os.path.basename(WAVE_FILE), 'RV wave file used')
    """
    if "MJDMID" not in header :
        try :
            header['MJDMID'] = header['MJDATE'] + (header['MJDEND'] - header['MJDATE'])/2.
        except :
            header['MJDMID'] = header['MJD-OBS'] + (header['JD-END'] - header['JD-OBS']) / 2.

    if "MJDMID" not in wheader :
        wheader['MJDMID'] = header['MJDMID']
    
    header['MJDATE'] = (header['MJDMID'], 'Modified Julian Date')

    header['RV_WAVTM'] = (wheader['MJDMID'],
                          'RV wave file time [mjd]')
    header['RV_WAVTD'] = (header['MJDMID'] - wheader['MJDMID'],
                          'RV timediff [days] btwn file and wave solution')
    """
    header['WFPDRIFT'] = (rv_drifts['WFPDRIFT'], 'Wavelength sol absolute CCF FP Drift [km/s]')
    header['RV_WAVFP'] = (rv_drifts['RV_WAVFP'], 'RV measured from wave sol FP CCF [km/s]')
    header['RV_SIMFP'] = (rv_drifts['RV_SIMFP'], 'RV measured from simultaneous FP CCF [km/s]')
    header['RV_DRIFT'] = (rv_drifts['RV_DRIFT'],
                          'RV drift between wave sol and sim. FP CCF [km/s]')
    header.set('ERVDRIFT', rv_drifts['RV_DRIFTERR'], 'Error of RV_DRIFT [km/s]')
    header['RV_OBJ'] = (props['MEAN_RV'],
                        'RV calc in the object CCF (non corr.) [km/s]')
    if type(rv_drifts['RV_DRIFT']) == float :
        header['RV_CORR'] = (props['MEAN_RV']-rv_drifts['RV_DRIFT'], 'RV corrected for FP CCF drift [km/s]')
    else :
        header['RV_CORR'] = ('None', 'RV corrected for FP CCF drift [km/s]')
    # ----------------------------------------------------------------------

    # work around to make old data compatible:
    if "EXTSN035" not in header.keys() :
        if "SNR" in header.keys():
            header["EXTSN035"] = header["SNR"]
        #else :
        #   header["EXTSN035"] = ??
        
    if output != "" :
        # log where we are writing the file to
        if verbose :
            print('Writing file to {0}'.format(output))
        
        # construct hdus
        hdu = fits.PrimaryHDU()
        t1 = fits.BinTableHDU(table1, header=header)
        t2 = fits.BinTableHDU(table2, header=header)
        # construct hdu list
        hdulist = fits.HDUList([hdu, t1, t2])
        # write hdulist
        hdulist.writeto(output, overwrite=True)
        props["file_path"] = output

    props["header"] = header
    props["RV_CCF"] = props['RV_CCF']
    props["MEAN_CCF"] = props['MEAN_CCF']

    return props


def apply_weights_to_ccf_mask(ccf_params, wl, flux, fluxerr, weight, median=True, remove_lines_with_nans=True, source_rv=0., verbose=False, plot=False) :
    
    sci_table = read_mask(ccf_params['MASK_FILE'], ccf_params['MASK_COLS'])
    
    if len(ccf_params['TELL_MASK_FILES']) :
        masktables = [sci_table]
        for m in ccf_params['TELL_MASK_FILES'] :
            masktables.append(read_mask(m, ccf_params['MASK_COLS']))
        masktable = vstack(masktables)
    else :
        masktable = sci_table

    # --------------------------------------------------------------------------
    # get mask centers, and weights
    lines_d, lines_wlc, lines_wei = get_mask(masktable, ccf_params["MASK_WIDTH"], ccf_params["MASK_MIN_WEIGHT"])

    lines_d = np.array(lines_d)
    lines_wlc = np.array(lines_wlc)
    lines_wei = np.array(lines_wei)

    mask_centers = np.array([])
    mask_weights = np.array([])
    mask_d = np.array([])

    speed_of_light_in_kps = constants.c / 1000.
    edge_size = ccf_params["CCF_WIDTH"]
    #edge_size = 7. # +-7 km/s ~ instrumental resolution
    
    median_flux = np.nanmedian(flux)
    mad = np.nanmedian(np.abs(flux - median_flux)) / 0.67449
    
    if verbose :
        print("median_flux={:.4f} sigma={:.4f}".format(median_flux, mad))
        
    minwl = wl[0] * (1.0 + edge_size / speed_of_light_in_kps)
    maxwl = wl[-1] * (1.0 - edge_size / speed_of_light_in_kps)
        
    keep_lines = lines_wlc > minwl
    keep_lines &= lines_wlc < maxwl

    keep_lines_wlc = lines_wlc[keep_lines]
    keep_lines_wei = lines_wei[keep_lines]
    keep_lines_d = lines_d[keep_lines]

    wl_ini = keep_lines_wlc * (1.0 + (source_rv - edge_size) / speed_of_light_in_kps)
    wl_end = keep_lines_wlc * (1.0 + (source_rv + edge_size) / speed_of_light_in_kps)

    nlines = 0
        
    if plot :
        wlc_starframe = keep_lines_wlc * (1.0 + source_rv / speed_of_light_in_kps)

        plt.vlines(wlc_starframe, median_flux - keep_lines_wei / np.nanmax(keep_lines_wei), median_flux, ls="--", lw=0.5, label="CCF mask lines")
        plt.plot(wl,flux,'-', lw=0.5, color="grey", label="Template spectrum")
        plt.plot(wl, 1.0 - fluxerr,'k:', lw=0.5, label="1-sigma noise threshold")

    for i in range(len(keep_lines_wlc)) :
        inline = (wl >= wl_ini[i]) & (wl <= wl_end[i])

        if len(flux[inline]) == 0 :
            continue
            
        line_mean_flux = np.nanmean(flux[inline])
        line_mean_fluxerr = np.nanmean(fluxerr[inline])
        line_mean_wl = np.nanmean(wl[inline])
            
        #if line_mean_flux > (1.0 - line_mean_fluxerr) :
        #    continue

        #print("Line: ",i,wl_ini[i],wl_end[i],line_mean_flux)
        if plot :
            p = plt.plot(wl[inline],flux[inline],'.',alpha=0.5)
            color = p[0].get_color()
            #plt.plot([np.mean(wl[inline])],[np.nanmean(weight[inline])],'o', color=color)
            plt.plot([line_mean_wl],[line_mean_flux],'o', color=color)
    
        if remove_lines_with_nans :
                
            if np.all(np.isfinite(flux[inline]) * np.isfinite(weight[inline])):
                mask_centers = np.append(mask_centers, keep_lines_wlc[i])
                mask_d = np.append(mask_d, keep_lines_d[i])
                    
                if median :
                    mask_weights = np.append(mask_weights, keep_lines_wei[i] * np.nanmedian(weight[inline]))
                else :
                    mask_weights = np.append(mask_weights, keep_lines_wei[i] * np.nanmean(weight[inline]))

                nlines += 1
        else :
            mask_centers = np.append(mask_centers, keep_lines_wlc[i])
            mask_d = np.append(mask_d, keep_lines_d[i])

            if np.isfinite(np.nanmedian(weight[inline])) :
                if median :
                    mask_weights = np.append(mask_weights, keep_lines_wei[i] * np.nanmedian(weight[inline]))
                else :
                    mask_weights = np.append(mask_weights, keep_lines_wei[i] * np.nanmean(weight[inline]))
            else :
                mask_weights = np.append(mask_weights, keep_lines_wei[i] * 0.)

            nlines += 1
    if verbose :
        print("Selected {0} lines".format(nlines))
    if plot :
        plt.show()

    outmask = {}

    outmask["centers"] = mask_centers
    outmask["weights"] = mask_weights
    outmask["widths"] = mask_d

    return outmask

# =============================================================================
# main routine -- version that inputs the spectrum in different format that
# assumes a pre-processing.
# =============================================================================
def run_ccf_eder(ccf_params, wave, fluxes, header, ccfmask, rv_drifts={}, filename="", berv=0., targetrv=0.0, normalize_ccfs=True, fit_type=0, output="", plot=False, verbose=False) :

    if rv_drifts == {} :
        rv_drifts["WFPDRIFT"] ='None'
        rv_drifts["RV_WAVFP"] = 'None'
        rv_drifts["RV_SIMFP"] = 'None'
        rv_drifts["RV_DRIFT"] = 0.
        rv_drifts["RV_DRIFTERR"] = 0.

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)

    blaze = np.ones_like(fluxes)
    
    # --------------------------------------------------------------------------
    # Photon noise uncertainty
    # --------------------------------------------------------------------------
    dkwargs = dict(spe=fluxes, wave=wave, sigdet=ccf_params["NOISE_SIGDET"],
                   size=ccf_params["NOISE_SIZE"], threshold=ccf_params["NOISE_THRES"])

    # run DeltaVrms2D
    dvrmsref, wmeanref = delta_v_rms_2d(**dkwargs)
    if verbose :
        print('Estimated RV uncertainty on spectrum is {0:.3f}'.format(wmeanref))

    # --------------------------------------------------------------------------
    # Calculate the CCF
    # --------------------------------------------------------------------------
    if verbose :
        print('\nRunning CCF calculation')

    props = ccf_calculation(ccf_params, wave, fluxes, blaze, targetrv, ccfmask["centers"],
                            ccfmask["weights"], berv, fit_type)

    # --------------------------------------------------------------------------
    # Calculate the mean CCF
    # --------------------------------------------------------------------------
    if verbose :
        print('\nRunning Mean CCF')
    props = mean_ccf(ccf_params, props, targetrv, fit_type, normalize_ccfs=normalize_ccfs, plot=plot)

    # --------------------------------------------------------------------------
    # Save file
    # --------------------------------------------------------------------------
    wheader = header

    props = write_file(props, filename, ccf_params['MASK_FILE'], header, wheader, rv_drifts, output=output, verbose=verbose)

    return props

# ==============================================================================
# Start CCF analysis
# ==============================================================================

def plot_ccfs(template_ccf) :

    rv = template_ccf["wl"]
    tccf = template_ccf["flux"]
    ccfs = template_ccf["flux_arr_sub"] * template_ccf["flux"]
    residuals = template_ccf["flux_arr_sub"]
    
    nspc = len(template_ccf["flux_arr_sub"])
    
    fig,ax = plt.subplots(nrows = 2, ncols = 1, sharex=True)
    
    for i in range(nspc):
        color = [i/nspc,1-i/nspc,1-i/nspc]
        ax[0].plot(rv, tccf, color = "green", lw=2, label="median CCF")
        ax[0].plot(rv, ccfs[i], color = color, alpha = 0.2)
        ax[1].plot(rv, residuals[i], color = color,alpha = 0.2)

    ax[0].set_title('Mean CCFs', fontsize=20)
    ax[0].set_xlabel('Velocity [km/s]', fontsize=20)
    ax[0].set_ylabel('CCF depth', fontsize=20)
    ax[0].tick_params(axis='both', which='major', labelsize=16)
    ax[0].tick_params(axis='both', which='minor', labelsize=12)

    ax[1].set_title('Residual CCFs', fontsize=20)
    ax[1].set_xlabel('Velocity [km/s]', fontsize=20)
    ax[1].set_ylabel('CCF residual depth', fontsize=20)
    ax[1].tick_params(axis='both', which='major', labelsize=16)
    ax[1].tick_params(axis='both', which='minor', labelsize=12)

    plt.tight_layout()
    #plt.legend()
#    plt.xticks(fontsize=16)
#    plt.yticks(fontsize=16)
    
    plt.show()


def calculate_rv_shifts(template_ccf, rvshifts, velocity_window, plot=False, verbose=False) :

    rvshifts_prev = deepcopy(rvshifts)

    rv = template_ccf["wl"]
    tccf = template_ccf["flux"]
    ccfs = template_ccf["flux_arr_sub"] * template_ccf["flux"]
    residuals = template_ccf["flux_arr_sub"]
    nspc = len(template_ccf["flux_arr_sub"])

    imin = np.argmin(tccf)
    
    if verbose :
        print("RV min detected at {:.4f} km/s".format(rv[imin]))
    
    window = np.abs(rv - rv[imin]) < velocity_window

    if verbose :
        print("Window: min_rv = {:.4f} km/s  and max_rv = {:.4f} km/s".format(rv[window][0],rv[window][-1]))

    corr_ccf = []
    for i in range(nspc) :
        spline = ius(rv, ccfs[i], ext=3, k=5)
        corr_ccf.append(spline(rv[window]+rvshifts[i]))
    corr_ccf = np.array(corr_ccf,dtype=float)
    
    deriv = np.gradient(tccf) / np.gradient(rv)
    deriv = deriv[window]
    deriv = deriv / np.nansum(deriv ** 2)
    
    per_ccf_rms = []

    if plot :
        plt.plot(rv, tccf, '-', lw=3, color="green")

    # pix scale expressed in CCF pixels
    #pix_scale = pixel_size_in_kps / np.nanmedian(np.gradient(rv))
    pix_scale = 1.0
    rvshifts_err = np.zeros_like(rvshifts)
    
    for i in range(nspc):
        residu = corr_ccf[i] - tccf[window]
        rms_resid = np.nanstd(residu)
        per_ccf_rms.append(rms_resid)
        
        rvshift = np.nansum(residu*deriv)
        rvshifts[i] -= rvshift
        
        # 1/dvrms -avoids division by zero
        inv_dvrms = deriv/(rms_resid * np.sqrt(pix_scale))
        rvshifts_err[i] = 1.0 / np.sqrt(np.nansum(inv_dvrms ** 2))

        if plot :
            plt.plot(rv[window]+rvshifts[i], corr_ccf[i])
        if verbose :
            print("Spectrum {} -> RV shift = {:.3f} m/s;  rms = {:.3f} m/s".format(i, rvshift*1000, rms_resid*1000))

    if plot :
        plt.show()

    rms_rv = np.nanstd(rvshifts_prev - rvshifts)
    if verbose :
        print("Final RV rms = {:.3f} m/s".format(rms_rv*1000))
    
    return rvshifts, rvshifts_err, rms_rv
    
    
def shift_ccf_data(rv, ccf_data, rvshifts, verbose=False) :

    dv = np.nanmedian(np.abs(rv[1:] - rv[:-1]))

    if verbose :
        print("Velocity sampling: dv = {:.4f} km/s".format(dv))

    nspc = len(rvshifts)
    
    min_rv, max_rv = -1e30,+1e30
    for i in range(nspc) :
        rel_rv = rv + rvshifts[i]
        locmin, locmax = np.nanmin(rel_rv), np.nanmax(rel_rv)
        
        if locmin > min_rv :
            min_rv = locmin
        if locmax < max_rv :
            max_rv = locmax

    if verbose :
        print("min_rv={:.4f} km/s, max_rv={:.4f} km/s".format(min_rv,max_rv))
    
    new_rv = []
    irv = min_rv + dv
    while irv < max_rv :
        new_rv.append(irv)
        irv+=dv
    new_rv = np.array(new_rv, dtype=float)

    new_ccf_data = []
    for i in range(nspc) :
        spline = ius(rv-rvshifts[i], ccf_data[i], ext=3, k=5)
        new_ccf_data.append(spline(new_rv))
    new_ccf_data = np.array(new_ccf_data,dtype=float)

    return new_rv, new_ccf_data
    
    
def ccf_analysis(rv, ccf_data, rvs, nsig_clip=0, velocity_window=10., maxiter=20, minimprov=1e-5, plot=False, verbose=False) :

    source_rv = np.nanmedian(rvs)
    
    #rvshifts = rvs - source_rv
    
    rvshifts = np.zeros_like(rvs)
    rvshiftvars = np.zeros_like(rvs)
    
    mod_rv, mod_ccf_data = deepcopy(rv), deepcopy(ccf_data)

    prev_rms_rv = 0

    for iter in range(maxiter) :

        # apply first shift the CCF data
        mod_rv, mod_ccf_data = shift_ccf_data(rv, ccf_data, rvshifts)
    
        # 1st pass - to build template from calibrated fluxes
        template_ccf = reduc_lib.calculate_template(mod_ccf_data, wl=mod_rv, fit=True, median=True, subtract=True, sub_flux_base=0.0, min_npoints=10, verbose=False, plot=False)

        # recover CCF data from template and residuals array
        red_ccf_data = template_ccf["flux_arr_sub"] + template_ccf["flux"]

        # 2nd pass - to build template from calibrated fluxes
        template_ccf = reduc_lib.calculate_template(red_ccf_data, wl=mod_rv, fit=True, median=True, subtract=True, sub_flux_base=0.0, verbose=False, plot=False)

        # apply sigma-clip using template and median dispersion in time as clipping criteria
        if nsig_clip > 0 :
            template_ccf = reduc_lib.sigma_clip(template_ccf, nsig=nsig_clip, interpolate=False, replace_by_model=False, sub_flux_base=0., plot=False)

        # recover CCF data from template and residuals array
        red_ccf_data = template_ccf["flux_arr_sub"] + template_ccf["flux"]

        # 3rd pass - Calculate a final template combined by the mean
        template_ccf = reduc_lib.calculate_template(red_ccf_data, wl=mod_rv, fit=True, median=False, subtract=False, sub_flux_base=1.0, verbose=False, plot=False)

        # calculate new shifts
        incr_rvshifts, rverrs, rms_rv = calculate_rv_shifts(template_ccf, np.zeros_like(rvshifts), velocity_window)

        rvshifts += incr_rvshifts
        rvshiftvars += rverrs*rverrs
        # calculate improvement with respect to previous fit
        improv = np.abs(prev_rms_rv - rms_rv)
        
        print("iter={} RV rms = {:.7f} m/s, an improvement of {:.7f} m/s ".format(iter, 1000*rms_rv, 1000*improv))

        if improv < minimprov :
            # plot final template product
            if plot :
                plot_ccfs(template_ccf)
            rvs = source_rv + rvshifts
            rverrs = np.sqrt(rvshiftvars)
            break

        #if np.abs(prev_rms_rv - rms_rv) < 1e-5
        prev_rms_rv = rms_rv

    return rvs, rverrs, template_ccf


def bisector(rv, ccf,  low_high_cut = 0.1, bottom_range=[0.10,0.40], top_range=[0.55,0.85], figure_title = '', doplot = False, ccf_plot_file = '', showplot=False):
    # use the props from the CCF determination code
    # Could be per-order or with the mean
    #rv = props['RV_CCF']
    #ccf = props['MEAN_CCF']

    # get minima
    imin = int(np.argmin(ccf))
    #print(imin,type(imin))

    # get point where the derivative changes sign at the edge of the line
    # the bisector is ambiguous passed this point
    width_blue =  imin - np.max(np.where(np.gradient(ccf[:imin])>0))
    #print(width_blue)
    width_red = np.min(np.where(np.gradient(ccf[imin:])<0))
    #print(width_red)

    # get the width from the side of the center that reaches
    # that point first
    width = int(np.min([width_blue, width_red]))

    # set depth to zero
    ccf -= np.min(ccf)

    # set continuum to one
    ccf /= np.min( ccf[ [imin - width, imin + width] ])

    # interpolate each side of the ccf slope at a range of depths
    depth = np.arange(low_high_cut,1-low_high_cut,0.001)

    # blue and red side of line
    g1 = (ccf[imin:imin - width:-1]>low_high_cut) & (ccf[imin:imin - width:-1]<(1-low_high_cut))
    spline1 = ius(ccf[imin:imin - width:-1][g1],rv[imin:imin - width:-1 ][g1], k=2)

    g2 = (ccf[imin : imin + width]>low_high_cut) & (ccf[imin : imin + width]<(1-low_high_cut))
    spline2 = ius(ccf[imin : imin + width][g2],rv[imin : imin + width][g2], k=2)

    # get midpoint
    bisector_position = (spline2(depth)+spline1(depth))/2

    # get bisector width
    width_ccf = (spline2(depth)-spline1(depth))

    # mean 'top' CCF between 55 and 85% of depth
    Vt = np.nanmean(bisector_position[(depth>top_range[0])*(depth<top_range[1])])
    Vt_ERR = np.nanstd(bisector_position[(depth>top_range[0])*(depth<top_range[1])])
    # mean 'bottom' CCF between 10% and 40% of depth
    Vb = np.nanmean(bisector_position[(depth>bottom_range[0])*(depth<bottom_range[1])])
    Vb_ERR = np.nanstd(bisector_position[(depth>bottom_range[0])*(depth<bottom_range[1])])

    Vs = Vt - Vb
    Vs_ERR = np.sqrt(Vt_ERR**2 + Vb_ERR**2)

    if doplot:
        # some nice plots
        plt.plot(rv[imin - width : imin+ width],ccf[imin - width : imin+ width],"k-", label = 'ccf')
        
        bottom = (ccf[imin - width : imin+ width] > bottom_range[0]) & (ccf[imin - width : imin+ width] < bottom_range[1])
        top = (ccf[imin - width : imin+ width] > top_range[0]) & (ccf[imin - width : imin+ width] < top_range[1])
        
        plt.plot(rv[imin - width : imin+ width][top],ccf[imin - width : imin+ width][top],"o",color="darkblue",label='Top',zorder=2)
        plt.plot(rv[imin - width : imin+ width][bottom],ccf[imin - width : imin+ width][bottom],"o",color="brown",label='Bottom',zorder=2)

        plt.plot(bisector_position,depth,label = 'bisector')
        plt.plot((bisector_position-np.mean(bisector_position))*100+np.mean(bisector_position),depth, label = 'bisector * 100')
        
        plt.legend()
        plt.title(figure_title,fontsize=16)
        plt.xlabel('Velocity (km/s)',fontsize=16)
        plt.ylabel('Depth',fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        if ccf_plot_file !='':
            plt.savefig(ccf_plot_file)
        if showplot :
            plt.show()

    # define depth in the same way as Perryman, 0 is top, 1 is bottom
    return 1-depth, bisector_position, width_ccf, Vs, Vs_ERR


def gauss(v,v0,ew,zp,amp):
    # gaussian with a constant offset. As we know that the ccfs are negative structures, amp will be negative
    return zp+amp*np.exp( -0.5*(v-v0)**2/ew**2)


def ccf_fwhm_and_biss(template_ccf, velocity_window=0, use_tight_window=False, plot=False, verbose=False) :

    rv = template_ccf["wl"]
    tccf = template_ccf["flux"]
    tccferr = template_ccf["fluxerr"]
    ccfs = template_ccf["flux_arr_sub"] * template_ccf["flux"]
    residuals = template_ccf["flux_arr_sub"]
    nspc = len(template_ccf["flux_arr_sub"])
            
    imin = np.argmin(tccf)

    if verbose :
        print("RV min detected at {:.4f} km/s".format(rv[imin]))

    depth, bis, width, bisspan, bisspan_err = bisector(rv, tccf, low_high_cut = 0.1, figure_title = 'CCF bisector analysis', doplot=plot, ccf_plot_file='', showplot=plot)

    # initialize window mask considering full CCF
    window = np.full_like(rv,True,dtype=bool)

    if use_tight_window :
        # get point where the derivative changes sign at the edge of the line
        width_blue =  imin - np.max(np.where(np.gradient(tccf[:imin])>0))
        width_red = np.min(np.where(np.gradient(tccf[imin:])<0))
        # initialize window mask
        window = np.full_like(rv,False,dtype=bool)
        # Select data within the window
        window[imin-width_blue:imin+width_red] = True
    else :
        if velocity_window :
            window = np.abs(rv - rv[imin]) < velocity_window/2
        pass
    
    tp0 = [rv[imin],1,1,-0.1]
    tfit, tpcov = curve_fit(gauss, rv[window], tccf[window], p0 = tp0)
    tfit_err = np.sqrt(np.diag(tpcov))

    if plot :
        plt.errorbar(rv[window], tccf[window], yerr=tccferr[window], fmt='o', color="green",zorder=1)
        plt.plot(rv[window],gauss(rv[window],tfit[0],tfit[1],tfit[2],tfit[3]),'-',color="red",zorder=2)

    fwhm, fwhmerr = [], []
    bis, biserr = [], []

    for i in range(nspc):
        residu = ccfs[i][window] - tccf[window]
        rms_resid = np.nanstd(residu)

        _, _, _, bisspan, bisspan_err = bisector(rv, ccfs[i])

        bis.append(bisspan)
        biserr.append(bisspan_err)

        p0 = deepcopy(tfit)
        fit, pcov = curve_fit(gauss, rv[window], ccfs[i][window], p0 = p0)
        fit_err = np.sqrt(np.diag(pcov))

        if verbose :
            print("Spectrum {} / {} -> gaussian fit: ".format(i+1,nspc))

        fwhm.append(2*np.sqrt(2*np.log(2)) * fit[1])
        fwhmerr.append(2*np.sqrt(2*np.log(2)) * fit_err[1])

        if plot :
            plt.plot(rv[window], ccfs[i][window], '-', alpha=0.3)

    bis, biserr = np.array(bis), np.array(biserr)
    fwhm, fwhmerr = np.array(fwhm), np.array(fwhmerr)
    
    if plot :
        plt.show()
    
    return fwhm, fwhmerr, bis, biserr


