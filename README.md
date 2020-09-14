# Dissertation_conglei


```{bash}
# directory of the Isca source code
export GFDL_BASE=/home/acq19ct/Isca
# "environment" configuration for Sheffield
export GFDL_ENV=sheffield-bessemer
# temporary working directory used in running the model
export GFDL_WORK=/home/acq19ct/Isca_work
# directory for storing model output
export GFDL_DATA=/home/acq19ct/Isca_data
```

```{bash}
diag.add_file('atmos_monthly', 30, 'days', time_units='days')
```



```{bash}
(isca_env)$cd $GFDL_BASE/exp/test_cases/frierson
(isca_env)$python frierson_test_case_second.py
```
```{bash}
root = 'data/'
nc_to_picture(root,1813)
Resize_Picture(1801)
```
