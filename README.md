# SOPHIE Tools
Toolkit to analyze s1d spectra obtained with the SOPHIE instrument at OHP. 

All the routines in this toolkit can be executed with the module `sophie_pipeline.py`. Here's an example to run the pipeline for a series of spectra obtained for the same object:

```
python sophie_pipeline.py --input=SOPHIE*s1d_A.fits -v
```

The command above will execute the following three modules, where each module generates a given product, as explained below.

```
* sophie_ccf_pipeline.py --> CCF radial velocity time series
* sophie_template.py  --> Template spectrum and calibrated spectra in the time series
* sophie_sindex.py --> s-index time series
```

Below are some examples to run each of these routines individually:
```
python $PATH/sophie_ccf_pipeline.py --input=SOPHIE*s1d_A.fits --ccf_mask=$PATH/masks/G2_nm.mas --output_rv_file=$OBJECTID_sophie_ccf.rdb -p -v

python $PATH/sophie_template.py --input=SOPHIE*s1d_A.fits --rv_file=$OBJECTID_sophie_ccf.rdb --output=$OBJECTID_sophie_template.fits -p -v

python $PATH/sophie_sindex.py --input=$OBJECTID_sophie_template.fits --output=$OBJECTID_sophie_sindex.txt -p -v
```
