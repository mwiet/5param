# 5param
Implementation of the 5-parameter stellar population synthesis model within the prospector inference framework.


Installation
-------------
All Python scripts are in Python 2.7. To create a Python 2.7 environment on linux with anaconda3, use:
```
conda create -n py27 python=2.7
```
The environment is activated by:
```
source activate py27
```
This environment will need to have the following modules installed:
- prospector
```
git clone https://github.com/bd-j/prospector
cd prospector
python setup.py install
```
- FSPS
```
pip install fsps
```
Might have to define $SPS_HOME directory before generating a stellar population:
```
export SPS_HOME="$HOME/fsps"
```
- sedpy
```
git clone https://github.com/bd-j/sedpy
cp /path/to/your/favorite/filters/*par sedpy/sedpy/data/filters/ 
cd sedpy 
python setup.py install
```

- emcee
```
pip install emcee
```

- multiprocessing
```
pip install multiprocessing
```

- corner (for diagnostic plots)
```
pip install corner
```
- pippi: follow the instructions given in https://github.com/patscott/pippi (Note: pippi requires ctioga2 which may not be possible to install on the ICL HPC)

- numpy, scipy (*may have to be set back to v.0.19*), pandas, matplotlib

Individual Run on ICL HPC
-------------
To run only one object, specify the parameters of the galaxy in the prospector[...]run.py file. This may be run locally or submitted as a job to the Imperial College London HPC queue using:
```
qsub run_[...].pbs
```
In the comments of the .pbs file, one can specify the nodes, cores, memory and time which will be used. 

Parallel Run on ICL HPC
-------------
In order to parallelise the job submission, use:
```
qsub run_[...]_par.pbs
```
which will call the respective prospector[...]run_par.py file. The .pbs file for a parallelised job has an additional parameter in its comments called -J which represents a range of indices for the number of parallel jobs. Make to give this in the following format:
```
#PBS -J 1-[# of parallel jobs/galaxies]
```
To check on the status of the job, type:
```
qstat -s
```
Each run will produce a .h5 file for diagnostic plots, a .h5py for pippi plots and a .csv file containing the maximum likelihood estimates and their uncertainties.

Photometric Data
-------------
The "photometric data" folder includes some of the raw photometric data from COSMOS2015 sample and SDSS which has been analysed.

Data
-------------
The "data" folder contains the parameters for the Mass-Metallicity prior as defined in Gallazzi et al., 2005.

Diagnostic Plots
-------------
There are two scripts in the "diagnostic plots" folder:
- plot_emcee.py will plot a simple triangle plot with the posterior distribution, a trace plot and the reconstructed SED for a particular galaxy when:
```
python plot_emcee.py [file_name].h5
```
- prospector_sed_variations.py allows to make plot SEDs and the relative change in the SEDs for different variations of parameters.

Pippi Scripts
-------------
The "pippi scripts" folder gives the .pip files which may be parsed into pippi in order to produce triangle plots from the .h5py files given in each run.

Analysis Plots
-------------
The "analysis plots" folder contains Python scripts in order to compare and analyse large samples of parameter estimates produced by the model. In order to read the parameters, they require to be in the same directory as the .csv files produced by prospector[...]run.py or prospector[...]run_par.py files.

Make sure that the file names defined in these scripts is the same as defined in the "outroot" parameter of the prospector[...]run.py or prospector[...]run_par.py files.

This folder also contains the .dat files including the Galaxy Stellar Mass Estimates from Davidzon et al., 2017, which can be graphed with the plot_mass_obs.py script.

Other Models
-------------
The folder named "other models" contains some model and running scripts of different parameter sets implemented in prospector. These are only there for completeness and may be inconsistent with some of the 5-parameter files in the main directory.
