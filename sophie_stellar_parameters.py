# -*- coding: iso-8859-1 -*-
"""
    Created on Dec 06 2022
    
    Description: Caculate atmospheric parameters from a template and time series of SOPHIE spectra
    
    @author: Eder Martioli <emartioli@lna.br>
    
    Laboratório Nacional de Astrofísica, Brazil.
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python /Volumes/Samsung_T5/Science/sophie/sophie_stellar_parameters.py --input=TOI-2141_template.fits -pv
    python /Volumes/Samsung_T5/Science/sophie/sophie_stellar_parameters.py --input=TOI-1736_template.fits -pv

    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys
import glob

import matplotlib.pyplot as plt
import reduc_lib
import spectrallib

import numpy as np

sophie_dir = os.path.dirname(__file__)




import warnings
warnings.filterwarnings('ignore')
import os, sys
import glob

import numpy as np
from numpy.polynomial.legendre import legval,legfit
from numpy import exp, loadtxt, pi, sqrt
from numpy.random import normal

from astropy.table import Table
from astropy.io import fits

from copy import deepcopy

from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

from multiprocessing import Pool
from multiprocessing import cpu_count
import concurrent.futures

from lmfit import Model
from lmfit.model import save_modelresult,load_modelresult
from lmfit.models import GaussianModel, ConstantModel, VoigtModel

from PyAstronomy import pyasl

#from reduc_lib import reduce_timeseries_of_spectra
#from sophielib import load_spectrum

from os import mkdir

import q2

import pandas as pd
pd.options.display.max_rows = 999

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import matplotlib as mpl
mpl.rcParams['text.usetex'] =False

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
from IPython.display import IFrame


def get_isoage_q2(name,tab,parameter_known='logg&plx'):
    tab_name= np.array([i.split('_')[0] for i in tab['name']])
    
    M = tab_name==name
    star = q2.Star(tab_name[M][0],
              teff=table_data_elem['teff'].to_numpy().astype(float)[M][0],
              logg=table_data_elem['logg'].to_numpy().astype(float)[M][0],
              feh=table_data_elem['M/H'].to_numpy().astype(float)[M][0])
    star.err_teff = table_data_elem['err_teff'].to_numpy().astype(float)[M][0]
    star.err_logg = table_data_elem['err_logg'].to_numpy().astype(float)[M][0]
    star.err_feh = table_data_elem['err_feh'].to_numpy().astype(float)[M][0]

    if parameter_known=='logg&plx':
        tab_simb = call_simbad(['%s'%name],display_table=None)
        Gmag = tab_simb['FLUX_G'].data.data[0]
        star.plx = tab_simb['PLX_VALUE'].data.data[0]
        star.err_plx = tab_simb['PLX_ERROR'].data.data[0]
        Vmag = Gmag*0.9940 +0.1929
        star.v =  Vmag
        star.err_v = 0.01
    #replace the output value in star.v and always use star.err_v = 0.01
    #this formula is valid only for solar twins!!
    sp = q2.isopars.SolvePars()
    sp.db = 'yy01.sql3'
    sp.feh_offset = -0.04
    sp.key_parameter_known = parameter_known
    sp.smooth_window_len_age = 23
    #pp.figure_format = 'pdf'
    


    pp = q2.isopars.PlotPars()
    pp.figure_format = 'pdf'
    pp.make_age_plot = True
    pp.age_xlim = [0, 14]
    pp.title_inside = '%s, Y$^2$'%star.name
    q2.isopars.solve_one(star, sp, pp)
    return star, sp
    
def call_simbad(stars,display_table=None):
    Simb = Simbad()
    Simb.add_votable_fields('id(HIP)','id(TOI)','main_id','coordinates','plx', 'plx_error',
                            'plx_bibcode','fluxdata(G)')
                            
    Simb.get_votable_fields()
    result_table = Simb.query_objects(stars)
    df = result_table.to_pandas()
    #df.to_csv('df_teste.csv')
    #for ii, lii in enumerate(result_table.keys()): print(ii,lii)
    result_table.show_in_notebook(display_length=500)
    if display_table is None:
        return result_table
    else:
        return result_table

def q2_param(name,EWfile_name,grid='odfnew',teff_i=5777,logg_i=4.44,feh_i=0.0,vt_i=1.,i=-1,step_teff_i=30,step_logg_i=0.3,step_vt_i=0.32):
    %matplotlib inline
    tab = Table.read(EWfile_name)
    print(tab.keys())
    print('%s_%s_hc'%(name,i))
    if np.any(np.array(tab.keys())=='%s_%s_hc'%(name,i))==True:
        pass
    else:
        tab.rename_column('%s(%s)'%(name,i), '%s_%s_hc'%(name,i))
    tab.write(EWfile_name,format='csv',overwrite=True)
    
    tab = Table()
    tab['id'] = np.array(['Moon','%s_%s_hc'%(name,i)])
    tab['teff'] = np.array([5777,teff_i])
    tab['logg'] = np.array([4.44,logg_i])
    tab['feh'] = np.array([0.0,feh_i])
    tab['vt'] = np.array([1,vt_i])
    
    
    tab.write('prov.csv',format='csv',overwrite=True)
    data = q2.Data('prov.csv',EWfile_name)
    
    Sun = q2.Star('Moon')
    star = q2.Star('%s_%s_hc'%(name,i))

    Sun.get_data_from(data)
    star.get_data_from(data)

    Sun.get_model_atmosphere(grid)
    star.get_model_atmosphere(grid)

    #print(star)
    #print(Sun)
    #input()
    star.teff=teff_i
    star.logg=logg_i
    star.feh=feh_i
    star.vt = vt_i

    sp=q2.specpars.SolvePars()
    sp.grid = grid
    sp.step_teff = step_teff_i
    sp.step_logg = step_logg_i
    sp.step_vt = step_vt_i
    sp.niter = 50
    sp.errors = True

    q2.specpars.solve_one(star, sp, Ref=Sun)
    display_parameters(star,sp,Ref=Sun)
    return star,data

def q2_outliers(name,data,i=-1,pplot='yes'):
    
    plt.figure(dpi=500,figsize=(9,6))
    file_FeI = 'DATA/%s_%s_hc_FeI.csv'%(name,i)
    file_FeII = 'DATA/%s_%s_hc_FeII.csv'%(name,i)

    paramFeI = np.loadtxt(file_FeI,delimiter=',')
    paramFeII = np.loadtxt(file_FeII,delimiter=',')
    
    mFeII = np.nanmean(paramFeII[:,3])
    sFeII = np.nanstd(paramFeII[:,3])
    M_FeII = abs(mFeII-paramFeII[:,3])<2*sFeII
    wav_FeII_outliers = paramFeII[:,0][~M_FeII]

    
    mFeI = np.nanmean(paramFeI[:,3])
    sFeI = np.nanstd(paramFeI[:,3])
    M_FeI = abs(mFeI-paramFeI[:,3])<2*sFeI
    wav_FeI_outliers = paramFeI[:,0][~M_FeI]
    if pplot=='yes':
        plt.subplot(211)
        plt.plot(paramFeI[:,1],paramFeI[:,3],'r*',alpha=0.2,label='Fe I')
        plt.plot(paramFeII[:,1],paramFeII[:,3],'bo',mfc='None',alpha=0.2,label='Fe II')
        #plt.plot(np.linspace(0,6,2),np.repeat(mFeI-2*sFeI,2),'r:')
        plt.plot(np.linspace(0,6,2),np.repeat(mFeI,2),'r-')
        #plt.plot(np.linspace(0,6,2),np.repeat(mFeI+2*sFeI,2),'r:')
        
        
        
        plt.ylim(-0.3,0.3)
        plt.legend(loc=0)
        plt.xlim(0,6)

        plt.subplot(212)
        plt.plot(paramFeI[:,2],paramFeI[:,3],'r*',alpha=0.2,label='Fe I')
        plt.plot(paramFeII[:,2],paramFeII[:,3],'bo',mfc='None',alpha=0.2,label='Fe II')
        #plt.plot(np.linspace(0,6,2),np.repeat(mFeI-2*sFeI,2),'r:')
        plt.plot(np.linspace(-5.9,-4.7,2),np.repeat(mFeI,2),'r-')
        #plt.plot(np.linspace(0,6,2),np.repeat(mFeI+2*sFeI,2),'r:')
                

        
        plt.plot(np.linspace(-5.9,-4.7,2),np.repeat(mFeII,2),'b-')
        

        
        plt.ylim(-0.3, 0.3)
        plt.xlim(-5.9,-4.7)
        plt.legend(loc=0)
        plt.tight_layout()
        plt.show()
    mask = np.array([])
    wavelength = np.array([])
    species = np.array([])
    ep = np.array([])
    gf = np.array([])
    Moon = np.array([])
    starname = np.array([])
    for ni, wav_i in enumerate(data.lines['wavelength']):
        if (np.any(wav_i==wav_FeI_outliers)==True) or (np.any(wav_i==wav_FeII_outliers)==True):
            print(ni+1,wav_i)
        else:
            wavelength = np.append(wavelength,data.lines['wavelength'][ni])
            species = np.append(species,data.lines['species'][ni])
            ep = np.append(ep,data.lines['ep'][ni])
            gf = np.append(gf,data.lines['gf'][ni])
            Moon = np.append(Moon,data.lines['Moon'][ni])
            starname = np.append(starname,data.lines['%s_%s_hc'%(name,i)][ni])
    tab=Table()
    tab['wavelength'] = wavelength
    tab['species'] = species
    tab['ep'] = ep
    tab['gf'] = gf
    tab['Moon'] =Moon
    tab['%s_%s_hc'%(name,i)] = starname
    stri_name = 'q2_tables/%s/%s_sigclip_%s_hc.csv'%(name,name,i)
    tab.write(stri_name,format='csv',overwrite=True)
    
    return stri_name
def define_line_list(line_list_file,SPECIES_SELECTED):
    line_list = np.genfromtxt(line_list_file,delimiter=',',dtype=str)
    q2_format = np.loadtxt('LINE_LIST/sun_2015_05_18.moog',dtype=str,usecols=(0,1,2,3,7))
    line_q2_vec = np.array([])
    for i, linha in enumerate(line_list):
        M=(abs(float(linha[0])-q2_format[:,0].astype(float))<=0.05)
        if np.any(M)==True:
            line_q2_vec = np.append(line_q2_vec,linha)
            line_q2_vec = np.append(line_q2_vec,q2_format[M])
    line_list = line_q2_vec.reshape(int(line_q2_vec.size/9.),9)
    for ss_ in np.unique(line_list[:,-1]):
        print('\t\t\t-> available species:',ss_)
    line_q2_vec = np.array([])
    for i, linha in enumerate(line_list):
        M= SPECIES_SELECTED==linha[-1]
        if np.any(M)==True:
            line_q2_vec = np.append(line_q2_vec,linha)
    line_list = line_q2_vec.reshape(int(line_q2_vec.size/9.),9)
    for ss_ in np.unique(line_list[:,-1]):
        print('\t\t\t\t-> selected species:',ss_)
    string_elements = ''.join( np.unique( np.array([spe[:2] for spe in np.unique(line_list[:,-1])])))
    print('\t\t\t\t-> string_elements =',string_elements)
    return line_list,string_elements
def derive_abundances(star,object_name,EWfile_name,species_ids):
    tab1 = Table()
    tab1['id']=np.array(['Moon',star.name])
    tab1['teff']=np.array([5777,star.teff])
    tab1['logg']=np.array([4.44,star.logg])
    tab1['feh']=np.array([0,star.feh])
    tab1['vt']=np.array([1,star.vt])
    tab1.write('TEMP/%s/prov.csv'%object_name,format='csv',overwrite=True)
    data = q2.Data('TEMP/%s/prov.csv'%object_name, EWfile_name)
    tab=Table.read('TEMP/%s/prov.csv'%object_name)
    display(tab)
    q2.abundances.get_all(data, output_file='TEMP/%s/prov2.csv'%object_name,species_ids=species_ids, reference='Moon', errors=True)
    tab=Table.read('TEMP/%s/prov2.csv'%object_name)
    display(tab)
    
    temp = list((star.name,round(star.teff,0),round(star.err_teff,0),round(star.logg,3),round(star.err_logg,3),round(star.feh,3),round(star.err_feh,3), star.converged,round(star.vt,3), round(star.err_vt,3)))
    for elemi in species_ids:
        temp.append(tab['[%s]'%elemi][tab['id']==star.name].data[0])
        temp.append(tab['e_[%s]'%elemi][tab['id']==star.name].data[0])
        temp.append(tab['n_[%s]'%elemi][tab['id']==star.name].data[0])

    return temp
def get_alpha_H_abundances(tab):
    alpha_elem = np.array([tab['MgI'].data , tab['SiI'].data , tab['TiI'].data, tab['TiII'].data]).astype(float)
    err_alpha_elem = np.array([tab['err_MgI'].data , tab['err_SiI'].data , tab['err_TiI'].data, tab['err_TiII'].data]).astype(float)
    alpha_H = np.nanmedian(alpha_elem,axis=0)
    err_alpha_H = np.sqrt( (np.nanstd(alpha_elem,axis=0))**2 + (np.nanmedian(err_alpha_elem)/np.sqrt(len(alpha_elem)))**2 )
    return alpha_H, err_alpha_H
def get_YMg_YAl_ratios(tab):
    YMg_ratio = tab['YII'].to_numpy().astype(float)-tab['MgI'].to_numpy().astype(float)
    err_YMg_ratio = abs(YMg_ratio)*np.sqrt( (tab['err_YII'].to_numpy().astype(float)/tab['YII'].to_numpy().astype(float))**2 + (tab['err_MgI'].to_numpy().astype(float)/tab['MgI'].to_numpy().astype(float))**2)

    YAl_ratio = tab['YII'].to_numpy().astype(float)-tab['AlI'].to_numpy().astype(float)
    err_YAl_ratio = abs(YAl_ratio)*np.sqrt( (tab['err_YII'].to_numpy().astype(float)/tab['YII'].to_numpy().astype(float))**2 + (tab['err_AlI'].to_numpy().astype(float)/tab['AlI'].to_numpy().astype(float))**2)

    return YMg_ratio, err_YMg_ratio,YAl_ratio, err_YAl_ratio
def get_MgSi_ratios(tab):
    MgSi_ratio = tab['MgI'].to_numpy().astype(float)-tab['SiI'].to_numpy().astype(float)
    err_MgSi_ratio = abs(MgSi_ratio)*np.sqrt( (tab['err_MgI'].to_numpy().astype(float)/tab['MgI'].to_numpy().astype(float))**2 + (tab['err_SiI'].to_numpy().astype(float)/tab['SiI'].to_numpy().astype(float))**2)
    return MgSi_ratio, err_MgSi_ratio
def get_MH(tab):#Salaris, Chieffi & Straniero (1993) alpha correction
    MH=tab['feh'].to_numpy().astype(float)+np.log10(0.638*10**(tab['alpha/Fe'].to_numpy().astype(float))+0.362)
    return MH
def chem_clocks(tab):
    age_YII_MgI = (0.204 - tab['YII/MgI'])/0.046
    err_age_YII_MgI = abs(tab['err_YII/MgI']/0.046)
    err_age_YII_MgI = np.sqrt(err_age_YII_MgI**2 + 1) # formal errors from s18
    
    age_YII_AlI =(0.231 - tab['YII/AlI'])/0.051
    err_age_YII_AlI = abs(tab['err_YII/AlI']/0.051)
    err_age_YII_AlI = np.sqrt(err_age_YII_AlI**2 + 0.9) # formal errors from s18
    return age_YII_MgI,err_age_YII_MgI,age_YII_AlI,err_age_YII_AlI


def display_parameters(star,sp,Ref):

    print("[Fe/H](Fe I)  = {0:5.3f} +/- {1:5.3f}".\
          format(star.iron_stats['afe1'], star.iron_stats['err_afe1']))
    print("[Fe/H](Fe II) = {0:5.3f} +/- {1:5.3f}".\
          format(star.iron_stats['afe2'], star.iron_stats['err_afe2']))
    print("A(FeI) vs. EP slope  = {0:.6f}".format(star.iron_stats['slope_ep']))
    print("A(FeI) vs. REW slope = {0:.6f}".format(star.iron_stats['slope_rew']))

    print("Final stellar parameters:")
    print("Teff = {0:4.0f} K, logg = {1:4.2f}, [Fe/H]= {2:5.2f}, vt = {3:4.2f} km/s".\
          format(star.teff, star.logg, star.feh, star.vt))
    
    q2.errors.error_one(star, sp, Ref)
    print("err_Teff = {0:2.0f} K, err_logg = {1:4.2f}, err_[Fe/H] = {2:4.2f}, err_vt = {3:4.2f}".\
      format(star.sp_err['teff'], star.sp_err['logg'], star.sp_err['afe'], star.sp_err['vt']))

def q2_param(name,EWfile_name,grid='odfnew',teff_i=5777,logg_i=4.44,feh_i=0.0,vt_i=1.,i=-1,step_teff_i=30,step_logg_i=0.3,step_vt_i=0.32):
    %matplotlib inline
    tab = Table.read(EWfile_name)
    print(tab.keys())
    print('%s_%s_hc'%(name,i))
    if np.any(np.array(tab.keys())=='%s_%s_hc'%(name,i))==True:
        pass
    else:
        tab.rename_column('%s(%s)'%(name,i), '%s_%s_hc'%(name,i))
    tab.write(EWfile_name,format='csv',overwrite=True)
    
    tab = Table()
    tab['id'] = np.array(['Moon','%s_%s_hc'%(name,i)])
    tab['teff'] = np.array([5777,teff_i])
    tab['logg'] = np.array([4.44,logg_i])
    tab['feh'] = np.array([0.0,feh_i])
    tab['vt'] = np.array([1,vt_i])
    
    
    tab.write('prov.csv',format='csv',overwrite=True)
    data = q2.Data('prov.csv',EWfile_name)
    
    Sun = q2.Star('Moon')
    star = q2.Star('%s_%s_hc'%(name,i))

    Sun.get_data_from(data)
    star.get_data_from(data)

    Sun.get_model_atmosphere(grid)
    star.get_model_atmosphere(grid)

    #print(star)
    #print(Sun)
    #input()
    star.teff=teff_i
    star.logg=logg_i
    star.feh=feh_i
    star.vt = vt_i

    sp=q2.specpars.SolvePars()
    sp.grid = grid
    sp.step_teff = step_teff_i
    sp.step_logg = step_logg_i
    sp.step_vt = step_vt_i
    sp.niter = 50
    sp.errors = True

    q2.specpars.solve_one(star, sp, Ref=Sun)
    display_parameters(star,sp,Ref=Sun)
    return star,data

def q2_outliers(name,data,i=-1,pplot='yes'):
    
    plt.figure(dpi=500,figsize=(9,6))
    file_FeI = 'DATA/%s_%s_hc_FeI.csv'%(name,i)
    file_FeII = 'DATA/%s_%s_hc_FeII.csv'%(name,i)

    paramFeI = np.loadtxt(file_FeI,delimiter=',')
    paramFeII = np.loadtxt(file_FeII,delimiter=',')
    
    mFeII = np.nanmean(paramFeII[:,3])
    sFeII = np.nanstd(paramFeII[:,3])
    M_FeII = abs(mFeII-paramFeII[:,3])<2*sFeII
    wav_FeII_outliers = paramFeII[:,0][~M_FeII]

    
    mFeI = np.nanmean(paramFeI[:,3])
    sFeI = np.nanstd(paramFeI[:,3])
    M_FeI = abs(mFeI-paramFeI[:,3])<2*sFeI
    wav_FeI_outliers = paramFeI[:,0][~M_FeI]
    if pplot=='yes':
        plt.subplot(211)
        plt.plot(paramFeI[:,1],paramFeI[:,3],'r*',alpha=0.2,label='Fe I')
        plt.plot(paramFeII[:,1],paramFeII[:,3],'bo',mfc='None',alpha=0.2,label='Fe II')
        #plt.plot(np.linspace(0,6,2),np.repeat(mFeI-2*sFeI,2),'r:')
        plt.plot(np.linspace(0,6,2),np.repeat(mFeI,2),'r-')
        #plt.plot(np.linspace(0,6,2),np.repeat(mFeI+2*sFeI,2),'r:')
        
        
        
        plt.ylim(-0.3,0.3)
        plt.legend(loc=0)
        plt.xlim(0,6)

        plt.subplot(212)
        plt.plot(paramFeI[:,2],paramFeI[:,3],'r*',alpha=0.2,label='Fe I')
        plt.plot(paramFeII[:,2],paramFeII[:,3],'bo',mfc='None',alpha=0.2,label='Fe II')
        #plt.plot(np.linspace(0,6,2),np.repeat(mFeI-2*sFeI,2),'r:')
        plt.plot(np.linspace(-5.9,-4.7,2),np.repeat(mFeI,2),'r-')
        #plt.plot(np.linspace(0,6,2),np.repeat(mFeI+2*sFeI,2),'r:')
                

        
        plt.plot(np.linspace(-5.9,-4.7,2),np.repeat(mFeII,2),'b-')
        

        
        plt.ylim(-0.3, 0.3)
        plt.xlim(-5.9,-4.7)
        plt.legend(loc=0)
        plt.tight_layout()
        plt.show()
    mask = np.array([])
    wavelength = np.array([])
    species = np.array([])
    ep = np.array([])
    gf = np.array([])
    Moon = np.array([])
    starname = np.array([])
    for ni, wav_i in enumerate(data.lines['wavelength']):
        if (np.any(wav_i==wav_FeI_outliers)==True) or (np.any(wav_i==wav_FeII_outliers)==True):
            print(ni+1,wav_i)
        else:
            wavelength = np.append(wavelength,data.lines['wavelength'][ni])
            species = np.append(species,data.lines['species'][ni])
            ep = np.append(ep,data.lines['ep'][ni])
            gf = np.append(gf,data.lines['gf'][ni])
            Moon = np.append(Moon,data.lines['Moon'][ni])
            starname = np.append(starname,data.lines['%s_%s_hc'%(name,i)][ni])
    tab=Table()
    tab['wavelength'] = wavelength
    tab['species'] = species
    tab['ep'] = ep
    tab['gf'] = gf
    tab['Moon'] =Moon
    tab['%s_%s_hc'%(name,i)] = starname
    stri_name = 'q2_tables/%s/%s_sigclip_%s_hc.csv'%(name,name,i)
    tab.write(stri_name,format='csv',overwrite=True)
    
    return stri_name
def define_line_list(line_list_file,SPECIES_SELECTED):
    line_list = np.genfromtxt(line_list_file,delimiter=',',dtype=str)
    q2_format = np.loadtxt('LINE_LIST/sun_2015_05_18.moog',dtype=str,usecols=(0,1,2,3,7))
    line_q2_vec = np.array([])
    for i, linha in enumerate(line_list):
        M=(abs(float(linha[0])-q2_format[:,0].astype(float))<=0.05)
        if np.any(M)==True:
            line_q2_vec = np.append(line_q2_vec,linha)
            line_q2_vec = np.append(line_q2_vec,q2_format[M])
    line_list = line_q2_vec.reshape(int(line_q2_vec.size/9.),9)
    for ss_ in np.unique(line_list[:,-1]):
        print('\t\t\t-> available species:',ss_)
    line_q2_vec = np.array([])
    for i, linha in enumerate(line_list):
        M= SPECIES_SELECTED==linha[-1]
        if np.any(M)==True:
            line_q2_vec = np.append(line_q2_vec,linha)
    line_list = line_q2_vec.reshape(int(line_q2_vec.size/9.),9)
    for ss_ in np.unique(line_list[:,-1]):
        print('\t\t\t\t-> selected species:',ss_)
    string_elements = ''.join( np.unique( np.array([spe[:2] for spe in np.unique(line_list[:,-1])])))
    print('\t\t\t\t-> string_elements =',string_elements)
    return line_list,string_elements
def derive_abundances(star,object_name,EWfile_name,species_ids):
    tab1 = Table()
    tab1['id']=np.array(['Moon',star.name])
    tab1['teff']=np.array([5777,star.teff])
    tab1['logg']=np.array([4.44,star.logg])
    tab1['feh']=np.array([0,star.feh])
    tab1['vt']=np.array([1,star.vt])
    tab1.write('TEMP/%s/prov.csv'%object_name,format='csv',overwrite=True)
    data = q2.Data('TEMP/%s/prov.csv'%object_name, EWfile_name)
    tab=Table.read('TEMP/%s/prov.csv'%object_name)
    display(tab)
    q2.abundances.get_all(data, output_file='TEMP/%s/prov2.csv'%object_name,species_ids=species_ids, reference='Moon', errors=True)
    tab=Table.read('TEMP/%s/prov2.csv'%object_name)
    display(tab)
    
    temp = list((star.name,round(star.teff,0),round(star.err_teff,0),round(star.logg,3),round(star.err_logg,3),round(star.feh,3),round(star.err_feh,3), star.converged,round(star.vt,3), round(star.err_vt,3)))
    for elemi in species_ids:
        temp.append(tab['[%s]'%elemi][tab['id']==star.name].data[0])
        temp.append(tab['e_[%s]'%elemi][tab['id']==star.name].data[0])
        temp.append(tab['n_[%s]'%elemi][tab['id']==star.name].data[0])

    return temp
def get_alpha_H_abundances(tab):
    alpha_elem = np.array([tab['MgI'].data , tab['SiI'].data , tab['TiI'].data, tab['TiII'].data]).astype(float)
    err_alpha_elem = np.array([tab['err_MgI'].data , tab['err_SiI'].data , tab['err_TiI'].data, tab['err_TiII'].data]).astype(float)
    alpha_H = np.nanmedian(alpha_elem,axis=0)
    err_alpha_H = np.sqrt( (np.nanstd(alpha_elem,axis=0))**2 + (np.nanmedian(err_alpha_elem)/np.sqrt(len(alpha_elem)))**2 )
    return alpha_H, err_alpha_H
def get_YMg_YAl_ratios(tab):
    YMg_ratio = tab['YII'].to_numpy().astype(float)-tab['MgI'].to_numpy().astype(float)
    err_YMg_ratio = abs(YMg_ratio)*np.sqrt( (tab['err_YII'].to_numpy().astype(float)/tab['YII'].to_numpy().astype(float))**2 + (tab['err_MgI'].to_numpy().astype(float)/tab['MgI'].to_numpy().astype(float))**2)

    YAl_ratio = tab['YII'].to_numpy().astype(float)-tab['AlI'].to_numpy().astype(float)
    err_YAl_ratio = abs(YAl_ratio)*np.sqrt( (tab['err_YII'].to_numpy().astype(float)/tab['YII'].to_numpy().astype(float))**2 + (tab['err_AlI'].to_numpy().astype(float)/tab['AlI'].to_numpy().astype(float))**2)

    return YMg_ratio, err_YMg_ratio,YAl_ratio, err_YAl_ratio
def get_MgSi_ratios(tab):
    MgSi_ratio = tab['MgI'].to_numpy().astype(float)-tab['SiI'].to_numpy().astype(float)
    err_MgSi_ratio = abs(MgSi_ratio)*np.sqrt( (tab['err_MgI'].to_numpy().astype(float)/tab['MgI'].to_numpy().astype(float))**2 + (tab['err_SiI'].to_numpy().astype(float)/tab['SiI'].to_numpy().astype(float))**2)
    return MgSi_ratio, err_MgSi_ratio
def get_MH(tab):#Salaris, Chieffi & Straniero (1993) alpha correction
    MH=tab['feh'].to_numpy().astype(float)+np.log10(0.638*10**(tab['alpha/Fe'].to_numpy().astype(float))+0.362)
    return MH
def chem_clocks(tab):
    age_YII_MgI = (0.204 - tab['YII/MgI'])/0.046
    err_age_YII_MgI = abs(tab['err_YII/MgI']/0.046)
    err_age_YII_MgI = np.sqrt(err_age_YII_MgI**2 + 1) # formal errors from s18
    
    age_YII_AlI =(0.231 - tab['YII/AlI'])/0.051
    err_age_YII_AlI = abs(tab['err_YII/AlI']/0.051)
    err_age_YII_AlI = np.sqrt(err_age_YII_AlI**2 + 0.9) # formal errors from s18
    return age_YII_MgI,err_age_YII_MgI,age_YII_AlI,err_age_YII_AlI


def display_parameters(star,sp,Ref):

    print("[Fe/H](Fe I)  = {0:5.3f} +/- {1:5.3f}".\
          format(star.iron_stats['afe1'], star.iron_stats['err_afe1']))
    print("[Fe/H](Fe II) = {0:5.3f} +/- {1:5.3f}".\
          format(star.iron_stats['afe2'], star.iron_stats['err_afe2']))
    print("A(FeI) vs. EP slope  = {0:.6f}".format(star.iron_stats['slope_ep']))
    print("A(FeI) vs. REW slope = {0:.6f}".format(star.iron_stats['slope_rew']))

    print("Final stellar parameters:")
    print("Teff = {0:4.0f} K, logg = {1:4.2f}, [Fe/H]= {2:5.2f}, vt = {3:4.2f} km/s".\
          format(star.teff, star.logg, star.feh, star.vt))
    
    q2.errors.error_one(star, sp, Ref)
    print("err_Teff = {0:2.0f} K, err_logg = {1:4.2f}, err_[Fe/H] = {2:4.2f}, err_vt = {3:4.2f}".\
      format(star.sp_err['teff'], star.sp_err['logg'], star.sp_err['afe'], star.sp_err['vt']))





parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="Spectral *s1d_A.fits data pattern",type='string',default="*.fits")
parser.add_option("-o", "--output", dest="output", help="Output s-index time series",type='string',default="")
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h sophie_spectral_analysis.py")
    sys.exit(1)

if options.verbose:
    print('Spectral s1d_A.fits data pattern: ', options.input)
    if options.output != "":
        print('Output s-index time series: ', options.output)

# Loading input template product
if options.verbose:
    print("Loading input template product ...")

template = spectrallib.read_template_product(options.input)

##################################
#### START SPECTRAL ANALYS #######
##################################
# extract fluxes within a certain spectral window and re-normalize data by a local continuum
wl, flux, fluxerr, fluxes, fluxerrs = spectrallib.extract_spectral_feature(template, wlrange=[388.8,402.0], cont_ranges=[[388.8,391.5],[394.6379,395.6379],[399.0,402.0]], polyn_order=6, plot=options.plot)


#basic inputs
wi = 3780
wf = 6950
dw = 0.01
dlc = 6 #angstrons spectral window around each line (as in Yana Galarza et al 2021)
wi_spectrograph = 4300 #sophie domain
wf_spectrograph = 6950 #sophie domain
SNR_lim = 50 #in the case of time series analysis
line_list_file = 'LINE_LIST/line_regions.csv'
SPECIES_SELECTED = np.array(['FeI', 'FeII','YII','AlI','MgI','TiI','TiII','SiI','CaI'])
resolving_power = 80000

name_vec=['TOI-1736','TOI-2141','HIP7585','HIP28066']

pandas_columns = ['name','teff','err_teff', 'logg','err_logg', 'feh', 'err_feh','converged', 'vt','err_vt']
SPECIES_SELECTED = ['FeI', 'FeII','YII','AlI','MgI','TiI','TiII','SiI','CaI']

for spe_i in SPECIES_SELECTED:
    pandas_columns.append('%s'%spe_i)
    pandas_columns.append('err_%s'%spe_i)
    pandas_columns.append('n_%s'%spe_i)
                  
table_data_elem = pd.DataFrame(data=None, columns=pandas_columns , dtype=str)


for name in name_vec:
    for i in range(-1,0):
        EWfile_name = 'q2_tables/%s/%s_%s_hc.csv'%(name,name,i)
        print(EWfile_name)
        exit()
        if len(glob.glob(EWfile_name))>0:
            star, data = q2_param(name,EWfile_name, grid='odfnew',
                                 teff_i=5777,logg_i=4.44,
                                 feh_i=0.0,vt_i=1.,i=i)

            EWfile_name = q2_outliers(name,data,i=i,pplot='yes')

            star, data = q2_param(name,EWfile_name, grid='odfnew',
                                 teff_i=star.teff,logg_i=star.logg,
                                 feh_i=star.feh,vt_i=star.vt,i=i,
                                 step_teff_i=10,step_logg_i=0.1,step_vt_i=0.1)

            if star.converged==True:
                temp = derive_abundances(star,name,EWfile_name,SPECIES_SELECTED)
                table_data_elem = table_data_elem.append(pd.DataFrame(data=[temp], columns= pandas_columns, dtype=str), ignore_index = True)
