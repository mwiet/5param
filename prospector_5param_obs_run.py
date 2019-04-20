from prospector_5param_obs_model import *

# THETA
run_params = {}
run_params["object_redshift"] = 0.7721
run_params['zred'] = run_params["object_redshift"]
run_params['objname'] = 581715
# SPS parameters
run_params["zcontinuous"] = 1

#Define filter sets
galex = ['galex_FUV', 'galex_NUV']
subaru = ['subaru_{0}'.format(b) for b in ['B','V','r','ip','zpp']]
ultravista = ['ultravista_{}'.format(b) for b in ['J', 'H', 'Ks']]
spitzer = ['spitzer_irac_ch'+n for n in ['1','2','3','4']]

# Dta parameters
run_params['filterset'] = subaru #+ ultravista + spitzer[:2]

run_params['massmet'] = False #Include mass-metallicity prior based on Gallazy et al., 2005
massmet = False
run_params['add_neb'] = False
run_params['add_dust'] =  False

verbose = False
run_params["verbose"] = False

print('Building model...')
model = load_model(**run_params)
print('Finished. Generating Stellar Population...')
sps = load_sps(**run_params)
print('Finished. Loading observations...')
obs = load_obs(**run_params)

theta = model.theta.copy()
title_text = ','.join(["{}={}".format(p, model.params[p][0]) for p in model.free_params])
print(theta)
print(model.free_params)

# Save mock spectrum and parameters in hdf5 file
from prospect.io import write_results
import h5py

print('Finished. Saving...')
if run_params['add_neb'] == False and run_params["add_dust"] == False:
    run_params["outfile"] = 'emcee_5param_obs'
elif run_params['add_neb'] == True and run_params["add_dust"] == False:
    run_params["outfile"] = 'emcee_5param_obs_neb'
elif run_params['add_neb'] == False and run_params["add_dust"] == True:
    run_params["outfile"] = 'emcee_5param_obs_dustem'
else:
    run_params["outfile"] = 'emcee_5param_obs_dustem+neb'

if run_params['massmet'] == True:
    run_params["outfile"] = run_params["outfile"] + '_massmet'
    
# Number of emcee walkers
run_params["nwalkers"] = 512
# Number of iterations of the MCMC sampling
run_params["niter"] = 2048
# Number of iterations in each round of burn-in
# After each round, the walkers are reinitialized based on the 
# locations of the highest probablity half of the walkers.
run_params["nburn"] = [128, 256, 512]
# The following number controls how often the chain is written to disk. This can be useful 
# to make sure that not all is lost if the code dies during a long MCMC run. It ranges 
# from 0 to 1; the current chains will be written out every `interval` * `niter` iterations.
# The default is 1, i.e. only write out at the end of the run.
run_params["interval"] = 0.30 # write out after every 30% of the sampling is completed.
outroot = "{0}_{1}".format(run_params['outfile'], run_params['objname'])
run_params['outroot'] = outroot
hfilename = outroot + '_5input_mcmc.h5'
hfile = h5py.File(hfilename, "a")
print("Writing to file {}".format(hfilename))
write_results.write_h5_header(hfile, run_params, model)
write_results.write_obs_to_h5(hfile, obs)

# --------
# RUN MINIMISER AND EMCEE
# --------

#print('Finished. Minimising initial parameters...')

from prospect.likelihood import chi_spec, chi_phot
def chivecfn(theta):
    """A version of lnprobfn that returns the simple uncertainty 
    normalized residual instead of the log-posterior, for use with 
    least-squares optimization methods like Levenburg-Marquardt.
    
    It's important to note that the returned chi vector does not 
    include the prior probability.
    """
    lnp_prior = model.prior_product(theta)
    if not np.isfinite(lnp_prior):
        return -np.infty

    # Generate mean model
    t1 = time.time()
    try:
        spec, phot, x = model.mean_model(theta, obs, sps=sps)
    except(ValueError):
        return -np.infty
    d1 = time.time() - t1

    chispec = chi_spec(spec, obs)
    chiphot = chi_phot(phot, obs)
    return np.concatenate([chispec, chiphot])

from prospect import fitting
#from scipy.optimize import least_squares
#run_params["nmin"] = 5
#run_params['ftol'] = 3e-16 
#run_params['maxfev'] = 5000
#run_params['xtol'] = 3e-16

# --- start minimization ----
#min_method = 'levenberg_marquardt'
#run_params["min_method"] = min_method

# We'll start minimization from "nmin" separate places, 
# the first based on the "init" values of each parameter and the 
# rest drawn from the prior.  This can guard against local minima.
#nmin = run_params["nmin"]
ts = time.time()  # time it
#pinitial = fitting.minimizer_ball(model.initial_theta.copy(), nmin, model)
guesses = []
#for i, pinit in enumerate(pinitial): #loop over initial guesses
#    res = least_squares(chivecfn, np.array(pinit), method='lm', x_scale='jac',
#                        xtol=run_params["xtol"], ftol=run_params["ftol"], 
#                        max_nfev=run_params["maxfev"])
#    guesses.append(res)

# Calculate chi-square of the results, and choose the best one
# fitting.reinitialize moves the parameter vector away from edges of the prior.
#chisq = [np.sum(r.fun**2) for r in guesses]
#best = np.argmin(chisq)
#theta_best = fitting.reinitialize(guesses[best].x, model,
#                                  edge_trunc=run_params.get('edge_trunc', 0.1))
initial_prob = None
pdur = time.time() - ts
#print(theta_best)
print('Finished. Initiating MCMC walk in {0} parameter space...'.format(str(model.free_params)))

# suppress output
fout = sys.stdout
fnull = open(os.devnull, 'w')
sys.stdout = fnull

# set the initial center of the ball of walkers to the best optimization result
#initial_center = theta_best.copy()
initial_center = model.initial_theta.copy()
print(initial_center)

# ----------
# Log Likelihood
# ----------
from prospect.likelihood import lnlike_spec, lnlike_phot, write_log
def lnprobfn(theta, nested=False, verbose=verbose, massmet=massmet):
    """
    Given a parameter vector, a dictionary of observational data 
    a model object, and an sps object, return the ln of the posterior. 
    This requires that an sps object (and if using spectra 
    and gaussian processes, a GP object) be instantiated.
    """

    # Calculate prior probability and exit if not within prior
    # Also if doing nested sampling, do not include the basic priors, 
    # since the drawing method includes the prior probability
    lnp_prior = model.prior_product(theta, nested=nested)
    if not np.isfinite(lnp_prior):
        return -np.infty
        
    # Generate "mean" model
    t1 = time.time()
    spec, phot, mfrac = model.mean_model(theta, obs, sps=sps)
    d1 = time.time() - t1
 
    # Calculate likelihoods
    t2 = time.time()
    lnp_spec = lnlike_spec(spec, obs=obs)
    lnp_phot = lnlike_phot(phot, obs=obs)
    d2 = time.time() - t2
    if verbose:
        write_log(theta, lnp_prior, lnp_spec, lnp_phot, d1, d2)
    if massmet:
        mass_ind = list(model.free_params).index('mass')
        met_ind = list(model.free_params).index('logzsol')
        lnp_massmet = lnmassmet(theta[mass_ind], theta[met_ind])
        return lnp_prior + lnp_phot + lnp_spec + lnp_massmet
    else:
        return lnp_prior + lnp_phot + lnp_spec

# Start sampling
from prospect import fitting
from multiprocessing import Pool
import os
import contextlib

os.environ["OMP_NUM_THREADS"] = "1"

from multiprocessing import cpu_count
ncpu = cpu_count()
print("{0} CPUs".format(ncpu))

if __name__ == '__main__':
    with contextlib.closing(Pool(processes=8)) as pool:
        tstart = time.time()  # time it
        out = fitting.run_emcee_sampler(lnprobfn, initial_center, model,
                                pool=pool, hdf5=hfile, **run_params)
        esampler, burn_loc0, burn_prob0 = out
        edur = time.time() - tstart

sys.stdout = fout

print('done emcee in {0}s'.format(edur))

write_results.write_hdf5(hfile, run_params, model, obs, 
                         esampler, guesses,
                         toptimize=pdur, tsample=edur,
                         sampling_initial_center=initial_center)

print('Finished. Saved to ' + str(hfilename))

time.sleep(3)
#Formatting Data for Pippi
pippi_format(hfilename)
print('Reformated to ' + str(hfilename[:-3])+ str('_pippi.hdf5'))

imax = np.argmax(esampler.lnprobability)
i, j = np.unravel_index(imax, esampler.lnprobability.shape)
theta_max = esampler.chain[i, j, :].copy()

from scipy import stats
chain = esampler.chain
flat = chain.reshape(-1, chain.shape[-1]).T
upper_list = []
lower_list = []
for i in range(len(theta_max)):
    p = flat[i,:]
    base_percentile = stats.percentileofscore(p, theta_max[i])
    if abs(base_percentile-50) < 16:
        upper = np.percentile(p, 84 + (base_percentile - 50))
        lower = np.percentile(p, 16 + (base_percentile - 50))
    else:
        if base_percentile-50 < 0:
            upper = np.percentile(p, 84 + base_percentile-50)
            lower = np.percentile(p, 0)
        elif base_percentile-50 > 0:
            upper = np.percentile(p, 100)
            lower = np.percentile(p, 16 + base_percentile-50)
    upper_list.append(upper)
    lower_list.append(lower)
    
import pandas as pd

df = pd.DataFrame(data = np.array([[run_params['objname']] + [run_params['zred']] + list(theta_max) + upper_list + lower_list]), columns = ['objname', 'zred'] + list(model.free_params) + [s + '_upper' for s in list(model.free_params)] + [s + '_lower' for s in list(model.free_params)])
df.to_csv(outroot + '_estimates.csv')
    
    