import time, sys, os
import numpy as np
from matplotlib.pyplot import *
import fsps
import sedpy
import prospect
from prospect.models import priors

# -----------
# Load Observations
# -----------

def load_obs(snr=10.0, add_noise=True, filterset=["sdss_g0", "sdss_r0"],
             **kwargs):
    """Make a mock dataset.  Feel free to add more complicated kwargs, and put
    other things in the run_params dictionary to control how the mock is
    generated.

    :param snr:
        The S/N of the phock photometry.  This can also be a vector of same
        lngth as the number of filters.

    :param add_noise: (optional, boolean, default: True)
        If True, add a realization of the noise to the mock spectrum
        
    :param filterset:
        A list of `sedpy` filter names.  Mock photometry will be generated 
        for these filters.
    """
    mock = {}
    mock['wavelength'] = None # No spectrum
    
    # Convert a model spectrum into broadband fluxes.
    from sedpy.observate import load_filters
    mock['filters'] = load_filters(filterset)
    mock["phot_wave"] = [f.wave_effective for f in mock["filters"]]

    # We need the models to make a mock
    sps = load_sps(**kwargs)
    model = load_model(**kwargs)
    params = {}
    for p in model.params.keys():
        if p in kwargs:
            params[p] = np.atleast_1d(kwargs[p])

    # And build the mock SED
    model.params.update(params)
    spec, phot, mfrac = model.mean_model(model.theta, mock, sps=sps)

    # Now store some output
    mock['true_spectrum'] = spec.copy()
    mock['true_maggies'] = phot.copy()
    from copy import deepcopy
    mock['mock_params'] = deepcopy(model.params)
    
    # And add noise
    pnoise_sigma = phot / snr
    if add_noise:
        pnoise = np.random.normal(0, 1, len(phot)) * pnoise_sigma
        mock['maggies'] = phot + pnoise
    else:
        mock['maggies'] = phot.copy()
    mock['maggies_unc'] = pnoise_sigma
    mock['mock_snr'] = snr
    mock['phot_mask'] = np.ones(len(phot), dtype=bool)

    # No spectrum
    mock['wavelength'] = None
    mock["spectrum"] = None
    
    return mock

# ----------------
# Stellar Population Synthesis
# ----------------
def load_sps(zcontinuous=1, **extras):
    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=zcontinuous)
    return sps

# -----------------
# Noise Model
# ------------------

def load_gp(**extras):
    return None, None

# --------------
# Conversion Functions
# --------------


def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
    return dust1_fraction*dust2



def zfrac_to_sfrac(z_fraction=None, **extras):
    """This transforms from latent, independent `z` variables to sfr
    fractions. The transformation is such that sfr fractions are drawn from a
    Dirichlet prior.  See Betancourt et al. 2010
    """

    sfr_fraction = np.zeros(len(z_fraction) + 1)
    sfr_fraction[0] = 1.0 - z_fraction[0]
    for i in range(1, len(z_fraction)): sfr_fraction[i] = np.prod(z_fraction[:i]) * (1.0 - z_fraction[i])
    sfr_fraction[-1] = np.clip(1 - np.sum(sfr_fraction[:-1]),0,1)
    return sfr_fraction



def zfrac_to_masses(logmass=None, z_fraction=None, agebins=None, **extras):
    """This transforms from latent, independent `z` variables to sfr fractions
    and then to bin mass fractions. The transformation is such that sfr
    fractions are drawn from a Dirichlet prior.  See Betancourt et al. 2010
    :returns masses:
        The stellar mass formed in each age bin.
    """
    # sfr fractions (e.g. Leja 2017)
    sfr_fraction = zfrac_to_sfrac(z_fraction)
    # convert to mass fractions
    time_per_bin = np.diff(10**agebins, axis=-1)[:,0]
    sfr_fraction *= np.array(time_per_bin)
    sfr_fraction /= sfr_fraction.sum()
    masses = 10**logmass * sfr_fraction
    return masses



def masses_to_zfrac(mass=None, agebins=None, **extras):
    """The inverse of zfrac_to_masses, for setting mock parameters based on
    real bin masses.
    """
    total_mass = mass.sum()
    time_per_bin = np.diff(10**agebins, axis=-1)[:,0]
    sfr_fraction = mass / time_per_bin
    sfr_fraction /= sfr_fraction.sum()
    z_fraction = np.zeros(len(sfr_fraction) - 1)
    z_fraction[0] = 1 - sfr_fraction[0]
    for i in range(1, len(z_fraction)):
        z_fraction[i] = 1.0 - sfr_fraction[i] / np.prod(z_fraction[:i])
    return total_mass, z_fraction

# This function transfroms from a fractional age of a burst to an absolute age.
# With this transformation one can sample in ``fage_burst`` without worry about
# the case tburst > tage.
def tburst_fage(tage=0.0, fage_burst=0.0, **extras):
    return tage * fage_burst

# Here is a really simple function that takes a **dict argument, picks out the
# `logzsol` key, and returns the value.  This way, we can have gas_logz find
# the value of logzsol and use it, if we uncomment the 'depends_on' line in the
# `gas_logz` parameter definition.
#
# One can use this kind of thing to transform parameters as well (like making
# them linear instead of log, or divide everything by 10, or whatever.) You can
# have one parameter depend on several others (or vice versa).  Just remember
# that a parameter with `depends_on` must always be fixed.

def stellar_logzsol(logzsol=0.0, **extras):
    return logzsol

# Reformat hdf5 outputs for Pippi
def pippi_format(filename, MPIrank=0):
    import json
    import h5py
    hf_origin = h5py.File(filename, 'r')
    sampling = hf_origin['sampling']
    chain = np.array(sampling['chain'])
  
    for k, v in sampling.attrs.items():
        if k == u'theta_labels':
            param_array = np.array(json.loads(v))
    
    lnprobability = np.array(sampling['lnprobability'])

    hf_new = h5py.File(filename[:-3] + str('_pippi.hdf5'))
    hf_new.create_group('sampling')
    ln_flat = lnprobability.flatten()
    hf_new['sampling'].create_dataset('LogLike', data = ln_flat)
    hf_new['sampling'].create_dataset('LogLike_isvalid', data = np.repeat(1, len(ln_flat)))
    hf_new['sampling'].create_dataset('MPIrank', data = np.repeat(MPIrank, len(ln_flat)))
    hf_new['sampling'].create_dataset('MPIrank_isvalid', data = np.repeat(1, len(ln_flat)))
    hf_new['sampling'].create_dataset('pointID', data = np.array(range(1, len(ln_flat)+1)))
    hf_new['sampling'].create_dataset('pointID_isvalid', data =  np.repeat(1, len(ln_flat)))
    for i in range(len(param_array)):
        v = chain[:,:,i]
        param = v.flatten()
        hf_new['sampling'].create_dataset(param_array[i], data=param)
        hf_new['sampling'].create_dataset(param_array[i] + '_isvalid', data = np.repeat(1, len(ln_flat)))
    hf_origin.close()
    hf_new.flush()
    hf_new.close()


# --------------
# MODEL_PARAMS
# --------------

def load_model(zred=0.0, add_neb=False, complex_atten=False, atten_bump=False, **extras):
    """Instantiate and return a ProspectorParams model subclass.
    
    :param zred: (optional, default: 0.1)
        The redshift of the model
        
    :param add_neb: (optional, default: False)
        If True, turn on nebular emission and add relevant parameters to the
        model.
    """
    from prospect.models.sedmodel import SedModel
    from prospect.models.templates import TemplateLibrary

    model_params = TemplateLibrary["alpha"]
    
    model_params["fagn"]["isfree"] = False
    model_params["agn_tau"]["isfree"] = False
    model_params["sfh"]["init"] = 0
    
    if add_neb == True:
        model_params['add_neb_emission']['init'] = True
    
    if complex_atten == True:
        # --- Complexify dust attenuation ---
        # Switch to Kriek and Conroy 2013
        model_params["dust_type"] = {'N': 1, 'isfree': False,
                                     'init': 4, 'prior': None}
    if atten_bump == True:
        # Slope of the attenuation curve, expressed as the index of the power-law
        # that modifies the base Kriek & Conroy/Calzetti shape.
        # I.e. a value of zero is basically calzetti with a 2175AA bump
        model_params["dust_index"] = {'N': 1, 'isfree': False,
                                     'init': 0.0, 'prior': None}


    return SedModel(model_params)




