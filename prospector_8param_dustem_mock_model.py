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

    model_params = TemplateLibrary["parametric_sfh"]
    
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


    # --- Distance ---
    model_params.update({'zred': {'N': 1,
                            'isfree': False,
                            'init': 0.1,
                            'units': '',
                            'prior': priors.TopHat(mini=0.0, maxi=4.0)}})

    # --- SFH --------
    # FSPS parameter
    model_params.update({'sfh': {'N': 1,
                            'isfree': False,
                            'init': 4,  # This is delay-tau
                            'units': 'type',
                            'prior': None}})

    model_params.update({'mass':{'N': 1, 'isfree': True,
                                'init': 1e10,
                                'init_disp': 1e9,
                                'units': r'M_\odot',
                                'prior': priors.LogUniform(mini=1e6, maxi=1e12)}})

    model_params.update({'logzsol': {'N': 1,
                            'isfree': True,
                            'init': -0.3,
                            'init_disp': 0.3,
                            'units': r'$\log (Z/Z_\odot)$',
                            'prior': priors.TopHat(mini=-2.0, maxi=0.19)}})

    # If zcontinuous > 1, use 3-pt smoothing
    model_params.update({'pmetals': {'N': 1,
                            'isfree': False,
                            'init': -99,
                            'prior': None}})

    # FSPS parameter
    model_params.update({'tau': {'N': 1,
                            'isfree': True,
                            'init': 1.0,
                            'init_disp': 0.5,
                            'units': 'Gyr',
                            'prior':priors.LogUniform(mini=0.101, maxi=100)}})

    # FSPS parameter
    model_params.update({'tage':{'N': 1,
                            'isfree': True,
                            'init': 5.0,
                            'init_disp': 3.0,
                            'units': 'Gyr',
                            'prior': priors.TopHat(mini=0.101, maxi=13.6)}})

    model_params.update({'fage_burst': {'N': 1,
                            'isfree': False,
                            'init': 0.0,
                            'units': 'time at wich burst happens, as a fraction of `tage`',
                            'prior': priors.TopHat(mini=0.9, maxi=1.0)}})

    # If we are going to be using emcee, it is useful to provide a 
    # minimum scale for the cloud of walkers (the default is 0.1)
    model_params["mass"]["disp_floor"] = 1e7
    model_params["mass"]["init_disp"] = 1e8
    model_params["tau"]["disp_floor"] = 1.0
    model_params["tage"]["disp_floor"] = 1.0


    # FSPS parameter
    model_params.update({'tburst': {'N': 1,
                            'isfree': False,
                            'init': 0.0,
                            'units': '',
                            'prior': None,}})
                            #'depends_on': tburst_fage}})  # uncomment if using bursts.

    # FSPS parameter
    model_params.update({'fburst': {'N': 1,
                            'isfree': False,
                            'init': 0.0,
                            'units': '',
                            'prior': priors.TopHat(mini=0.0, maxi=0.5)}})
    
    # ------  IMF  ---------

    model_params.update({'imf_type': {'N': 1,
                                 'isfree': False,
                                 'init': 1, #1 = chabrier
                                 'units': None,
                                 'prior': None}})

    # --- Dust ---------
    # FSPS parameter
    model_params.update({'dust_type': {'N': 1,
                            'isfree': False,
                            'init': 0,  # power-laws
                            'units': 'index',
                            'prior': None}})
    # FSPS parameter
    model_params.update({'dust2': {'N': 1,
                            'isfree': True,
                            'init': 0.35,
                            'reinit': True,
                            'init_disp': 0.3,
                            'units': 'Diffuse dust optical depth towards all stars at 5500AA',
                            'prior': priors.TopHat(mini=0.0, maxi=2.0)}})

    # FSPS parameter
    model_params.update({'dust1': {'N': 1,
                            'isfree': False,
                            'init': 0.0,
                            'units': 'Extra optical depth towards young stars at 5500AA',
                            'prior': None}})

    # FSPS parameter
    model_params.update({'dust_tesc': {'N': 1,
                            'isfree': False,
                            'init': 7.0,
                            'units': 'definition of young stars for the purposes of the CF00 dust model, log(Gyr)',
                            'prior': None}})

    # FSPS parameter
    model_params.update({'dust_index': {'N': 1,
                            'isfree': False,
                            'init': -0.7,
                            'units': 'power law slope of the attenuation curve for diffuse dust',
                            'prior': priors.TopHat(mini=-1.0, maxi=0.4)}})

    # FSPS parameter
    model_params.update({'dust1_index': {'N': 1,
                            'isfree': False,
                            'init': -1.0,
                            'units': 'power law slope of the attenuation curve for young-star dust',
                            'prior': None}})
    # ---- Dust Emission ---
    # FSPS parameter
    model_params.update({'add_dust_emission': {'N': 1,
                            'isfree': False,
                            'init': True,
                            'units': 'index',
                            'prior': None}})

    # An example of the parameters controlling the dust emission SED.  There are others!
 
    model_params.update({'duste_gamma': {'N': 1,
                            'isfree': True,
                            'init': 0.01,
                            'init_disp': 0.2,
                            'disp_floor': 0.15,
                            'units': None,
                            'prior': priors.TopHat(mini=0.0, maxi=1.0)}})


    model_params.update({'duste_umin': {'N': 1,
                            'isfree': True,
                            'init': 1.0,
                            'init_disp': 5.0,
                            'disp_floor': 4.5,
                            'units': None,
                            'prior': priors.TopHat(mini=0.1, maxi=25.0)}})

    model_params.update({'duste_qpah': {'N': 1,
                            'isfree': True,
                            'init': 2.0,
                            'init_disp': 3.0,
                            'disp_floor': 3.0,
                            'units': 'percent',
                            'prior': priors.TopHat(mini=0.0, maxi=7.0)}})

    # --- Stellar Pops ------------
    # One could imagine changing these, though doing so *during* the fitting will
    # be dramatically slower.
    # FSPS parameter
    model_params.update({'tpagb_norm_type': {'N': 1,
                            'isfree': False,
                            'init': 2,
                            'units': 'index',
                            'prior': None}})

    # FSPS parameter
    model_params.update({'add_agb_dust_model': {'N': 1,
                            'isfree': False,
                            'init': True,
                            'units': 'index',
                            'prior': None}})

    # FSPS parameter
    model_params.update({'agb_dust': {'N': 1,
                            'isfree': False,
                            'init': 1,
                            'units': 'index',
                            'prior': None}})

    # --- Nebular Emission ------

    # For speed we turn off nebular emission in the demo
    model_params.update({'add_neb_emission': {'N': 1,
                            'isfree': False,
                            'init': False,
                            'prior': None}})
    if add_neb == True:
        model_params['add_neb_emission']['init'] = True
        
    # FSPS parameter
    model_params.update({'gas_logz': {'N': 1,
                            'isfree': False,
                            'init': 0.0,
                            'units': r'log Z/Z_\odot',
    #                        'depends_on': stellar_logzsol,
                            'prior': priors.TopHat(mini=-2.0, maxi=0.5)}})

    # FSPS parameter
    model_params.update({'gas_logu': {'N': 1,
                            'isfree': False,
                            'init': -2.0,
                            'units': '',
                            'prior': priors.TopHat(mini=-4, maxi=-1)}})

    # --- Calibration ---------
    # Only important if using a NoiseModel
    model_params.update({'phot_jitter': {'N': 1,
                            'isfree': False,
                            'init': 0.0,
                            'units': 'mags',
                            'prior': priors.TopHat(mini=0.0, maxi=0.2)}})


    return SedModel(model_params)




