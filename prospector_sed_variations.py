from prospector_5param_mock_model import *

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


# THETA
run_params = {}

# SPS parameters
run_params["zcontinuous"] = 1



# Input parameters for the mock spectrum
run_params['mass'] = 1e8
run_params['logzsol'] = -0.5
run_params['tage'] = 10.0
run_params['tau'] = 0.5
run_params['dust2'] = 0.6
run_params['zred'] = 0.1
run_params['add_neb'] = False
run_params["add_dust"] = False

#Define filter sets
galex = ['galex_FUV', 'galex_NUV']
subaru = ['subaru_{0}'.format(b) for b in ['B','V','r','ip','zpp']]
ultravista = ['ultravista_{}'.format(b) for b in ['J', 'H', 'Ks']]
spitzer = ['spitzer_irac_ch'+n for n in ['1','2','3','4']]

# Mock data parameters
run_params['snr']= 20.0
run_params['add_noise'] = True
run_params['filterset'] = subaru + ultravista + spitzer[:2]

verbose = False
run_params["verbose"] = False

#var_param = 'mass'
#label = r'$M$'
#unit = r'$M_{\odot}$'
#var = [8e7, 1e8, 1.2e8]

#var_param = 'logzsol'
#label = r'$log_{10}(Z/Z_{\odot})$'
#unit = r''
#var = [-1.8, 0.0, 0.1]

var_param = 'tau'
label = r'$\tau$'
unit = r'Gyrs'
var = [0.2, 5.0, 10.0]

#var_param = 'tage'
#label = r'$t_{age}$'
#unit = r'Gyrs'
#var = [0.2, 5.0, 10.0]

#var_param = 'dust2'
#label = r'$\tau_{d}$'
#unit = r'diffuse dust optical depth at 5500 angstroms'
#var = [0.0, 0.6, 1.0]
spec = []
wave = []
maximum = []
minimum = []
figure(figsize=(16,8))

xmin, xmax = 1e3, 1e6
for param in var:
    run_params[var_param] = param
    print(run_params[var_param])
    
    print('Building model...')
    model = load_model(**run_params)
    params = {}
    for p in model.params.keys():
        if p in run_params:
            params[p] = np.atleast_1d(run_params[p])
    model.params.update(params)
    print('Finished. Generating Stellar Population...')
    sps = load_sps(**run_params)
    print('Finished. Loading observations...')
    obs = load_obs(**run_params)
    
    a = 1.0 + model.params.get('zred', 0.0) # cosmological redshifting
    # photometric effective wavelengths
    wphot = obs["phot_wave"]
    # spectroscopic wavelengths
    if obs["wavelength"] is None:
        # *restframe* spectral wavelengths, since obs["wavelength"] is None
        wspec = sps.wavelengths.copy()
        wspec *= a #redshift them
    else:
        wspec = obs["wavelength"]
    
    wave.append(wspec)
        
    # Generate the model SED at some value of theta
    theta = model.theta.copy()
    print(theta)
    print('Generating mock SED...')
    initial_spec, initial_phot, initial_mfrac = model.sed(theta, obs=obs, sps=sps)
    spec.append(initial_spec)
    sub_spec = initial_spec[(wspec > xmin) & (wspec < xmax)]
    maximum.append(max(sub_spec))
    minimum.append(min(sub_spec))

loglog(wave[0], spec[0], label=r'{0} = {1} {2}'.format(label, var[0], unit), lw=0.7, alpha=0.7)
loglog(wave[1], spec[1], label=r'{0} = {1} {2}'.format(label, var[1], unit), lw=0.7, alpha=0.7)
loglog(wave[2], spec[2], label=r'{0} = {1} {2}'.format(label, var[2], unit), lw=0.7, alpha=0.7)
ylim([min(minimum)+min(minimum)*0.01, max(maximum)+ max(maximum)])

xlim([xmin, xmax])
xlabel('Wavelength [angstroms]')
ylabel('Flux Density [maggies]')
legend(loc='best', fontsize=20)
tight_layout()
savefig('SED_comparison_' + str(var_param) + '.png')
close()

figure(figsize=(16,8))
maximum = []
minimum = []
for i in range(len(var)):
    diff = np.array(spec[i])-np.array(spec[1])
    if i != 1:
        change = diff/abs(np.array(spec[1]))
        plot(wave[i], change, label=r'{0} = {1} {2}'.format(label, var[i], unit), lw=0.7, alpha=0.7)
        sub_spec = change[(wave[i] > xmin) & (wave[i] < xmax)]
        maximum.append(max(sub_spec))
        minimum.append(min(sub_spec))
    else:
        plot(wave[i], np.zeros(len(wave[i])), label=r'{0} = {1} {2}'.format(label, var[i], unit), lw=0.7, alpha=0.7)
    
plot(0,0, markersize=0)
xscale('log')
yscale('symlog')
xlim([xmin, xmax])
ymin = min(minimum)*2
ymax = max(maximum)*2
ymin_order = int(np.log10(abs(ymin)))
ymax_order = np.ceil(np.log10(abs(ymax)))
yt = (-np.logspace(0, ymin_order, abs(ymin_order + 1))).tolist() + [0.0]  + np.logspace(0, ymax_order, abs(ymax_order+1)).tolist()
ylim([-10**ymin_order, 10**ymax_order])
ylim([-10**1, 10**0])
yt = [-10, -1, 0 , 1]
yticks(yt, label = yt)
xlabel(r'Wavelength [angstroms]')
ylabel(r'$(F-F_{0})/F_{0}$')
legend(loc='best', fontsize=20)
tight_layout()
savefig('SED_relative_diff_' + str(var_param) + '.png')