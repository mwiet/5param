from prospector_8param_dustem_obs_model import *

# THETA
run_params = {}
run_params["object_redshift"] = 0.1015
run_params['zred'] = run_params["object_redshift"]
run_params['objname'] = 576667

# SPS parameters
run_params["zcontinuous"] = 1

#Define filter sets
galex = ['galex_FUV', 'galex_NUV']
subaru = ['subaru_{0}'.format(b) for b in ['B','V','r','ip','zpp']]
ultravista = ['ultravista_{}'.format(b) for b in ['J', 'H', 'Ks']]
spitzer = ['spitzer_irac_ch'+n for n in ['1','2','3','4']]

# Mock data parameters

run_params['filterset'] = galex + subaru + ultravista + spitzer[:2]

# Input parameters for the mock spectrum
run_params['add_neb'] = False
run_params["add_dust"] = True #ALREADY TRUE IN MODEL FILE


verbose = False
run_params["verbose"] = False

print('Building model...')
model = load_model(**run_params)
print('Finished. Generating Stellar Population...')
sps = load_sps(**run_params)
print('Finished. Loading observations...')
obs = load_obs(**run_params)

# Generate the model SED at some value of theta
theta = model.theta.copy()
title_text = ','.join(["{}={}".format(p, model.params[p][0]) for p in model.free_params])

# Save mock spectrum and parameters in hdf5 file
from prospect.io import write_results
import h5py

print('Finished. Saving...')
if run_params['add_neb'] == False and run_params["add_dust"] == False:
    run_params["outfile"] = 'emcee_5param_obs'
elif run_params['add_neb'] == True and run_params["add_dust"] == False:
    run_params["outfile"] = 'emcee_5param_mobs_neb'
elif run_params['add_neb'] == False and run_params["add_dust"] == True:
    run_params["outfile"] = 'emcee_8param_obs_dustem'
else:
    run_params["outfile"] = 'emcee_8param_obs_dustem+neb'
outroot = "{0}_{1}".format(run_params['outfile'], run_params['objname'])
run_params['outroot'] = outroot
hfilename = outroot + '.h5'
hfile = h5py.File(hfilename, "a")
print("Writing to file {}".format(hfilename))
write_results.write_h5_header(hfile, run_params, model)
write_results.write_obs_to_h5(hfile, obs)

# --------
# RUN EMCEE
# --------

print('Finished. Initiating MCMC walk...')
# Number of emcee walkers
run_params["nwalkers"] = 128
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

# suppress output
fout = sys.stdout
fnull = open(os.devnull, 'w')
sys.stdout = fnull

# set the initial center of the ball of walkers to the best optimization result
initial_center = model.initial_theta.copy()
guesses=[]
pdur = 0

# ----------
# Log Likelihood
# ----------
from prospect.likelihood import lnlike_spec, lnlike_phot, write_log
def lnprobfn(theta, nested=False, verbose=verbose):
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

    return lnp_prior + lnp_phot + lnp_spec

# Start sampling
from prospect import fitting
tstart = time.time()  # time it
out = fitting.run_emcee_sampler(lnprobfn, initial_center, model,
                                pool=None, hdf5=hfile, **run_params)
esampler, burn_loc0, burn_prob0 = out
edur = time.time() - tstart

sys.stdout = fout

print('done emcee in {0}s'.format(edur))

write_results.write_hdf5(hfile, run_params, model, obs, 
                         esampler, guesses,
                         toptimize=pdur, tsample=edur,
                         sampling_initial_center=initial_center)

print('Finished. Saved to ' + str(hfilename))

#Formatting Data for Pippi
pippi_format(hfilename)
print('Reformated to ' + str(hfilename[:-3])+ str('_pippi.hdf5'))