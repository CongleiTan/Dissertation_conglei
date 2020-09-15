# Dissertation_conglei
To start with, you’ll need a copy of the source code and there are two ways to for followers
to realize it. You can fork the Isca repository to your own github username, or clone directly
from the ExeClim group.
```{bash}
$ git clone https://github.com/ExeClim/Isca
$ cd Isca
```
In the second step, you can find the python module in the src directory and you can
install it using pip. We recommend that you can the Anaconda distribution and create an
environment to do this (in the code below called ”isca env”)
```{bash}
$ conda create -n isca_env python ipython
...
$ source activate isca_env
(isca_env)$ cd src/extra/python
(isca_env)$ pip install -r requirements.txt
```
After that, you can install the isca python module in ”development mode”. This will
allow you, if you want, to edit the src/extra/python/isca files and have those changes be
used when you next run an Isca script.
```{bash}
(isca_env)$ pip install -e .
```
Compiling for the first time

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
At University of Sheffield, Isca is compiled using:
• compilers/intel/15.0.3
• mpi/intel/openmpi/1.10.0
• libs/netcdf/4.3.2
• mpif90
• mpicc
• h5pfc
Before Isca is compiled, an environment is first configured which loads the specific compilers
and libraries essential to build the code. This done by setting the environment variable
GFDL ENV in your session.
For example, on the EMPS workstations at Sheffield, I have done the following on my HPC
```{bash}
diag.add_file('atmos_monthly', 30, 'days', time_units='days')
```



```{bash}
(isca_env)$cd $GFDL_BASE/exp/test_cases/frierson
(isca_env)$python frierson_test_case.py
```
```{bash}
root = 'data/'
nc_to_picture(root,1813)
Resize_Picture(1801)
```

```{bash}
train_and_test(root_dir,seed)
```
