import time, sys, os
import numpy as np
from matplotlib.pyplot import *

switch_backend('agg')

# re-defining plotting defaults
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
rcParams.update({'xtick.major.pad': '7.0'})
rcParams.update({'xtick.major.size': '7.5'})
rcParams.update({'xtick.major.width': '1.5'})
rcParams.update({'xtick.minor.pad': '7.0'})
rcParams.update({'xtick.minor.size': '3.5'})
rcParams.update({'xtick.minor.width': '1.0'})
rcParams.update({'ytick.major.pad': '7.0'})
rcParams.update({'ytick.major.size': '7.5'})
rcParams.update({'ytick.major.width': '1.5'})
rcParams.update({'ytick.minor.pad': '7.0'})
rcParams.update({'ytick.minor.size': '3.5'})
rcParams.update({'ytick.minor.width': '1.0'})
rcParams.update({'xtick.color': 'k'})
rcParams.update({'ytick.color': 'k'})
rcParams.update({'font.size': 30})
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'


from prospect.io.read_results import results_from, get_sps
from prospect.io.read_results import traceplot, subcorner

print('A model is being assumed dynamically... Please check if correct. Assuming:')



from sys import argv
outroot = str(argv[1])

if '5param_mock' in outroot:
    from prospector_5param_mock_model import *
    print('5 parameters with mock data')
elif '5param_obs' in outroot:
    from prospector_5param_obs_model import *
    print('5 parameters with observations')
elif '8param_mock_dustem' in outroot:
    from prospector_8param_dustem_mock_model import *
elif '8param_obs_dustem' in outroot:
    from prospector_8param_dustem_mock_model import *
    print('8 parameters (dust emission) with observations')
elif '5param_massmet_mock' in outroot:
    from prospector_5param_massmet_mock_model import *
    print('5 parameters (mass-metallicity rel.) with mock data')
elif '_alpha_' in outroot and 'mock' in outroot:
    from prospector_alpha_mock_model import *
    print('Prospector-alpha from Leja et al. 2017 with mock data')
else:
    from prospector_5param_mock_model import *
    print('5 parameters with mock data')

print(outroot) # This is the start of the filename where we saved the results

# grab results (dictionary), the obs dictionary, and our corresponding models
# When using parameter files set `dangerous=True`
res, obs, mod = results_from("{}".format(outroot), dangerous=True)
run_params = res['run_params']

# let's look at what's stored in the `res` dictionary
print('-------')
print('Stored info:')
print(res.keys())
print('')
if 'mock' in outroot:
    theta_input = np.squeeze(np.array([obs["mock_params"][p] for p in res['theta_labels']]))
    print(theta_input)
    #Parameter Traces
    chosen = np.random.choice(run_params["nwalkers"], 
                          size=10, replace=False)
else:
    theta_input = None
    chosen = np.random.choice(run_params["nwalkers"], 
                          size=10, replace=False)
tracefig = traceplot(res, figsize=(20,10), chains=chosen)
savefig(outroot + '_parameter_traces.png')


# Conerplot

# Maximum Likelihood (of the locations visited by the MCMC sampler)
imax = np.argmax(res['lnprobability'])
i, j = np.unravel_index(imax, res['lnprobability'].shape)
theta_max = res['chain'][i, j, :].copy()
n_param = len(theta_max)
size = int(n_param*n_param) + 2


print('Max Likelihood value: {}'.format(theta_max))
cornerfig = subcorner(res, start=0, thin=5, truths=theta_input,
                      fig=subplots(n_param,n_param,figsize=(size,size))[0])
savefig(outroot + '_cornerplot.png')

# randomly chosen parameters from chain
randint = np.random.randint
nwalkers, niter = run_params['nwalkers'], run_params['niter']
theta = res['chain'][randint(nwalkers), randint(niter)]
res['run_params']['param_file'] = 'plot_emcee.py'

# generate models
model = load_model(**res['run_params'])
sps = load_sps(**res['run_params'])  # this works if using parameter files
mspec_map, mphot_map, _ = model.mean_model(theta_max, obs, sps=sps)
a = 1.0 + model.params.get('zred', 0.0)
# photometric effective wavelengths
wphot = obs["phot_wave"]
# spectroscopic wavelengths
if obs["wavelength"] is None:
    # *restframe* spectral wavelengths, since obs["wavelength"] is None
    wspec = sps.wavelengths
    wspec *= a #redshift them
else:
    wspec = obs["wavelength"]

# Make plot of data and model
close()
figure(figsize=(16,8))

loglog(wspec, mspec_map, label='Fitted spectrum (Max Like)',
       lw=0.7, color='green', alpha=0.7)
errorbar(wphot, mphot_map, label='Fitted photometry (Max Like)',
         marker='s', markersize=10, alpha=0.8, ls='', lw=3, 
         markerfacecolor='none', markeredgecolor='green', 
         markeredgewidth=3)
errorbar(wphot, obs['maggies'], yerr=obs['maggies_unc'], 
         label='Observed photometry', ecolor='red', 
         marker='o', markersize=10, ls='', lw=3, alpha=0.8, 
         markerfacecolor='none', markeredgecolor='red', 
         markeredgewidth=3)

xmin, xmax = 1e3, 1e6
ymin, ymax = 1e-16, 1e-8

xlabel(r'Wavelength [angstroms]')
ylabel('Flux Density [maggies]')
xlim([xmin, xmax])
ylim([ymin, ymax])
legend(loc='best', fontsize=20)
tight_layout()
savefig(outroot + '_final_SED.png')