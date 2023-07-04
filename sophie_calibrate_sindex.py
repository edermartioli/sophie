# -*- coding: iso-8859-1 -*-
"""
    Created on Nov 11 2022
    
    Description: Calibrate s-index of SOPHIE spectra to the MW system
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python /Volumes/Samsung_T5/Science/sophie/sophie_calibrate_sindex.py --input=TOI-2141_template.fits -pv

    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys
import glob


from astropy.io import fits
from astropy.wcs import WCS

import matplotlib.pyplot as plt
import numpy as np
import spectrallib
import pandas as pd
from scipy.optimize import minimize
from scipy import stats

import emcee
import corner

sophie_dir = os.path.dirname(__file__)
calibrators_dir = os.path.join(sophie_dir, "s-index_calibrators/")
calibrators_ref_table = os.path.join(calibrators_dir, "SMW_Sindex.csv")

def extract_spectral_feature(wl, flux, fluxerr, wlrange=[], cont_ranges=[], polyn_order=6, divide_by_deltamag=False, plot=False) :

    if wlrange == [] :
        wlrange = [wl[0],wl[-1]]

    keep = (wl > wlrange[0]) & (wl < wlrange[1])
    keep &= (np.isfinite(flux)) & (np.isfinite(fluxerr))
    
    wl, flux, fluxerr = wl[keep], flux[keep], fluxerr[keep]
        
    cont = wl < 0
    for r in cont_ranges :
        cont ^= (wl > r[0]) & (wl < r[-1])

    coeffs = spectrallib.fit_continuum(wl[cont], flux[cont], function='polynomial', order=polyn_order, nit=5, rej_low=1., rej_high=4.0, grow=1, med_filt=1, percentile_low=0., percentile_high=100., min_points=100, xlabel="wavelength", ylabel="flux", return_polycoeffs=True, plot_fit=False, verbose=False)

    continuum = np.poly1d(coeffs)(wl)

    if plot :
        #plt.errorbar(wl, flux, yerr=fluxerr, fmt='.', lw=0.3, alpha=0.3, zorder=1, label="template spectrum")
        plt.plot(wl, flux, '-', lw=2, alpha=0.8, zorder=1, label="template spectrum")
        plt.plot(wl, continuum, '--', lw=2, zorder=2, label="continuum")

    flux /= continuum
    fluxerr /= continuum
    
    if plot :
        plt.legend(fontsize=18)
        plt.xlabel(r"$\lambda$ [nm]", fontsize=20)
        plt.ylabel(r"Flux", fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.show()

    return wl, flux, fluxerr


#vamos definir a funcao que eu acho que se comporta como o observavel
def func(tempo, A, B):
    valor = A*tempo + B
    return valor



def log_likelihood(theta, x, y, xerr, yerr):
    m, b = theta
    model = m * x + b
    sigma2 = xerr**2 + yerr**2 + model**2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def log_prior(theta):
    m, b = theta
    if -5.0 < m < 10. and -10.0 < b < 10.0 :
        return 0.0
    return -np.inf
    
def log_probability(theta, x, y, xerr, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, xerr, yerr)


parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="Input list of calibrators",type='string',default="")
parser.add_option("-o", "--output", dest="output", help="Output s-index time series",type='string',default="")
parser.add_option("-e", action="store_true", dest="use_errors", help="use_errors", default=False)
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h sophie_calibrate_sindex.py")
    sys.exit(1)

# Loading input template product
if options.verbose:
    print("Loading spectra of calibrators ...")

if options.input == "" :
    options.input = os.path.join(calibrators_dir,"list_of_calibrators.txt")
list_of_calibrators = np.genfromtxt(options.input,dtype=str)

obj_spectra = {}
obj_meansindex, obj_stdsindex = np.array([]), np.array([])

for i in range(len(list_of_calibrators)):
    
    obj_spectra[list_of_calibrators[i]] = sorted(glob.glob("{}/{}/*.fits".format(calibrators_dir,list_of_calibrators[i])))

    print("Calibrator {} of {} : {}".format(i+1,len(list_of_calibrators),list_of_calibrators[i]))

    inputdata = obj_spectra[list_of_calibrators[i]]

    s_indexes = np.array([])

    for j in range(len(inputdata)):
    
        basename = os.path.basename(inputdata[j])
        hdu = fits.open(inputdata[j])
        header = hdu[0].header
        wcs = WCS(header)
        index = np.arange(header['NAXIS1'])
        
        #bjd = header["HIERARCH OHP DRS BJD"]
        #rv_source = header["HIERARCH OHP DRS BERV"]
        
        flux = hdu["S1D_A"].data
        keep = (flux>0) & (np.isfinite(flux))
        fluxerr = np.full_like(flux,np.nan)
        
        fluxerr[keep] = np.sqrt(flux[keep])
        wavelength = wcs.wcs_pix2world(index[:,np.newaxis], 0)
        wavelength = wavelength.flatten()
            
        factor = 10**10
        wavelength = wavelength*factor
        wl = wavelength/10
            
            
        if options.use_errors :
            wl, flux, fluxerr = extract_spectral_feature(wl[keep], flux[keep], fluxerr[keep], wlrange=[388.8,402.0], cont_ranges=[[388.8,391.5],[394.6379,395.6379],[399.0,402.0]], polyn_order=6, plot=False)
            sidx, esidx = spectrallib.sindex_montecarlo(wl, flux, fluxerr, deltalamca=0.1, deltalamcont=1.0, lamh=393.368, lamk=396.849, lamv=390.107, lamr=400.107, verbose=True, plot=False)
            print("Spectrum {} of {} --> {} for calibrator {}:  s-index = {:.4f} +/- {:.4f}".format(j+1,len(inputdata),basename,list_of_calibrators[i],sidx,esidx))
        else :
            sidx = spectrallib.sindex(wl, flux, deltalamca=0.1, deltalamcont=1.0, lamh=393.368, lamk=396.849, lamv=390.107, lamr=400.107, verbose=options.verbose, plot=False)
            print("Spectrum {} of {} --> {} for calibrator {}:  s-index = {:.4f}".format(j+1,len(inputdata),basename,list_of_calibrators[i],sidx))
            
        s_indexes = np.append(s_indexes, sidx)
        
    obj_meansindex = np.append(obj_meansindex, np.nanmedian(s_indexes))
    obj_stdsindex = np.append(obj_stdsindex, stats.median_absolute_deviation(s_indexes, nan_policy='omit'))


smw = pd.read_csv(calibrators_ref_table)
erromax = 1000
keep = (smw['SMW']>0) & (obj_stdsindex < erromax)

Smedio = np.array(obj_meansindex[keep])
Smw = np.array(smw['SMW'][keep])
smwstd = smw['SStd'][keep]
sstd = obj_stdsindex[keep]

ssophie = np.array(Smedio)
smw = np.array(Smw)

np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = np.array([1., 0.])

soln = minimize(nll, initial, args=(Smedio, smw, sstd, smwstd))
m_ml, b_ml = soln.x

print("Maximum likelihood estimates:")
print("m = {0:.4f}".format(m_ml))
print("b = {0:.4f}".format(b_ml))


pos = soln.x + 1e-4 * np.random.randn(32, 2)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(Smedio, smw, sstd, smwstd)
)
sampler.run_mcmc(pos, 5000, progress=True);

fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["m", "b"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
plt.show()

tau = sampler.get_autocorr_time()
print("Integrated autocorrelation time: ".format(tau))

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

coeffs, ecoeffs = [], []
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    print("{} = {:.4f} + {:.4f} - {:.4f} ".format(labels[i], mcmc[1], q[0], q[1]))
    coeffs.append(mcmc[1])
    ecoeffs.append((q[0]+q[1])/2)

fig = corner.corner(flat_samples, show_titles=True, labels=labels, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84], truths=[coeffs[0], coeffs[1]], labelsize=12, labelpad=2.0)
plt.show()


plt.errorbar(Smedio, smw, xerr=sstd, yerr=smwstd, fmt=".k", capsize=0)
x0 = np.linspace(np.min(Smedio)-0.2, np.max(Smedio)+0.2, 500)

inds = np.random.randint(len(flat_samples), size=100)
for ind in inds:
    sample = flat_samples[ind]
    plt.plot(x0, np.dot(np.vander(x0, 2), sample[:2]), "C1", alpha=0.1)

plt.plot(x0, coeffs[0] * x0 + coeffs[1], "k", alpha=0.3, lw=3, label="MCMC")
plt.plot(x0, np.dot(np.vander(x0, 2), [m_ml, b_ml]), ":k", label="ML")
plt.legend(fontsize=14)
#plt.xlim(0, 10)
plt.xlabel(r'<s-index> [SOPHIE]', {'color': 'k','fontsize': 20})
plt.ylabel(r'<S-index> [MW]', {'color': 'k','fontsize': 20})
plt.show()


"""
params, params_covariance = optimize.curve_fit(func, Smedio, Smw, p0=[1, 1])
print(params)
x = np.linspace(0.15,0.85,10)

funcao = func(x, params[0], params[1])
funcaonova = func(x, params[0]-0.31, params[1]-0.013)
#funcao = func(x, 0.12, 0.13)
erros = np.sqrt(np.diag(params_covariance))

fig, ax = plt.subplots(figsize=[6, 5.5])

ssophie = np.array(Smedio)
smw = np.array(Smw)
nomes = list_of_calibrators

for i in range(len(ssophie)):
    plt.text(ssophie[i]+0.001,smw[i],nomes[i],size=10,zorder=3,color='r')

plt.errorbar(ssophie,smw, xerr = sstd, yerr=smwstd, linestyle='None', color='k', zorder=-1, capsize=0,alpha=1)
plt.plot(ssophie, smw,'ko')
plt.plot(x,funcao,'k-')
#plt.plot(x,funcaonova,'b-')
plt.xlabel(r'<s-index> [SOPHIE]', {'color': 'k','fontsize': 20})
plt.ylabel(r'<S-index> [MW]', {'color': 'k','fontsize': 20})

plt.text(0.15, 0.65, r'$S_{MW} =$ ('+str(np.round(params[0],3))+r'$\pm$'+str(np.round(erros[0],3))+r') $S_{SOPHIE}$ + ('+str(np.round(params[1],3))+r'$\pm$'+str(np.round(erros[1],3))+r')',size=10)

#plt.xlim(0.1,0.7)
#plt.ylim(0.1,0.7)

plt.minorticks_on()
plt.tick_params(axis='both',which='minor', direction = "in",top = True,right = True, length=5,width=1,labelsize=15)
plt.tick_params(axis='both',which='major', direction = "in",top = True,right = True, length=8,width=1,labelsize=15)
plt.tight_layout()

plt.show()
"""
