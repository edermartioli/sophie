# SOPHIE Tools
Toolkit to analyze s1d spectra obtained with the SOPHIE instrument at OHP. 

All the routines in this toolkit can be executed with the module `sophie_pipeline.py`. Here's an example to run the pipeline for a series of spectra obtained for the same object:

```
python sophie_pipeline.py --input=SOPHIE*s1d_A.fits -pv
```

The command above will execute the following three modules, where each module generates a given product, as explained below.

```
* sophie_ccf_pipeline.py --> CCF radial velocity time series
* sophie_template.py  --> Template spectrum and calibrated spectra in the time series
* sophie_sindex.py --> s-index time series

python /Volumes/Samsung_T5/Science/sophie/sophie_ccf_pipeline.py --input=SOPHIE*s1d_A.fits --ccf_mask=/Volumes/Samsung_T5/Science/sophie/masks/G2_nm.mas --output_rv_file=TOI2141_sophie_ccf.rdb -p -v

python /Volumes/Samsung_T5/Science/sophie/sophie_template.py --input=SOPHIE*s1d_A.fits --rv_file=TOI2141_sophie_ccf.rdb --output=TOI2141_sophie_template.fits -p -v

python /Volumes/Samsung_T5/Science/sophie/sophie_sindex.py --input=TOI2141_sophie_template.fits --output=TOI2141_sophie_sindex.txt -p -v
```

 
