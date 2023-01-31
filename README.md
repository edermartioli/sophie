# SOPHIE Tools

Toolkit for analyzing e2ds or s1d spectra obtained with the SOPHIE instrument at OHP.

All routines in this toolkit can be run with the sophie_pipeline.py module. Here is an example to run the pipeline for a series of spectra obtained for the same object:

```
python sophie_pipeline.py --input=SOPHIE*e2ds_A.fits -v
```

The above command will execute the following modules, where each module generates a certain product, as explained below.

```
* sophie_ccf_pipeline.py --> CCF radial velocity time series + time series of the bisector span and FWHM
* sophie_template.py  --> Template spectrum and calibrated spectra in the time series
* sophie_sindex.py --> s-index time series
* sophie_halpha.py --> H-alpha time series
* sophie_NaI.py --> Na I time series
* sophie_CaI.py --> Ca I (insensitive to activity) time series
* sophie_compile_products.py --> Combine all times series data above into a single big table
* sophie_timeseries_analysis.py --> basic time series analysis of the data in the big table
```

Below are some examples for running each of these routines individually:

```
python $PATH/sophie_ccf_pipeline.py --input=SOPHIE*e2ds_A.fits --ccf_mask=$PATH/masks/G2_nm.mas --output_rv_file=$OBJECTID_sophie_ccfrv.rdb --output_bis_file=$OBJECTID_sophie_ccfbis.rdb --output_fwhm_file=$OBJECTID_sophie_ccffwhm.rdb  -p -v

python $PATH/sophie_template.py --input=SOPHIE*e2ds_A.fits --rv_file=$OBJECTID_sophie_ccf.rdb --output=$OBJECTID_sophie_template.fits -p -v -n

python $PATH/sophie_sindex.py --input=$OBJECTID_sophie_template.fits --output=$OBJECTID_sophie_sindex.txt -p -v
```
