import pandas as pd
import numpy as np
from matplotlib.pyplot import *
from scipy.odr import *
import matplotlib
from matplotlib.ticker import FuncFormatter 

switch_backend('agg')
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

C15 = pd.read_csv('c15_clean_264.csv')
z_whole = pd.read_csv('C15_clean.csv')['ZPDF']
m_list = []
z_list = []
m_upper = []
m_lower = []
remove_list = []
for index, row in C15.iterrows():
    i = int(row['NUMBER'])
    try:
        params = pd.read_csv('_5bands_emcee_5param_obs_massmet_' + str(i) + '_estimates.csv') #Change names according to outroot defined in 'prospector_[...]_run.py' file (currently only plotting params)
        params2 = pd.read_csv('emcee_5param_obs_' + str(i) + '_estimates.csv') 
        params3 = pd.read_csv('_5bands_emcee_5param_obs_' + str(i) + '_estimates.csv')
        m_list.append(float(params['mass']))
        m_upper.append(float(params['mass_upper']))
        m_lower.append(float(params['mass_lower']))
        z_list.append(float(params['zred']))
    except:
        print('Did not find ' + str(i))
        remove_list.append(i)
    
    
       
m_list = np.log10(np.array(m_list))
m_upper = np.log10(np.array(m_upper))
m_lower = np.log10(np.array(m_lower))
m_err = [m_list-m_lower, m_upper-m_list]

for i in remove_list:
    C15 = C15[C15['NUMBER'] != i]

print('Plotting ' + str(len(C15)) + ' galaxies')
m_c15 = np.array(C15['MASS_BEST'])
upper_diff = np.array(C15['MASS_MED_MAX68']) - np.array(C15['MASS_MED'])
m_c15_upper = np.array(C15['MASS_BEST']) + upper_diff
lower_diff = np.array(C15['MASS_MED_MIN68']) - np.array(C15['MASS_MED'])
m_c15_lower = np.array(C15['MASS_BEST']) + lower_diff
m_c15_err = [m_c15-m_c15_lower, m_c15_upper-m_c15]

#Linear fitting to mass vs mass plot in order to quantify systematic errors
def f(B, x):
    '''Linear function y = m*x + b'''
    # B is a vector of the parameters.
    # x is an array of the current x values.
    # x is in the same format as the x passed to Data or RealData.
    #
    # Return an array in the same format as y passed to Data or RealData.
    return B[0]*x + B[1]

linear = Model(f)
mydata = RealData(m_c15, m_list, sx=np.array(m_c15_err).mean(axis=0), sy=np.array(m_err).mean(axis=0))
myodr = ODR(mydata, linear, beta0=[1., 0.])
myoutput = myodr.run()
myoutput.pprint()

#Plotting mass estimate vs previous estimate
rcParams["figure.figsize"] = [8,7]
cmap = matplotlib.cm.get_cmap('jet')
norm = matplotlib.colors.Normalize(vmin=min(z_list), vmax=max(z_list))

errorbar(m_c15, m_list, xerr = m_c15_err, yerr = m_err, fmt = 'o', markersize=0, linewidth=0.1, capsize=4.0, capthick = 0.1, c='grey')
scatter(m_c15, m_list, c=np.log10(z_list), cmap=cmap, s=2.0)

def fmt(x, pos):
    z = 10**(x)
    return r'${}$'.format(round(z,3))

colorbar(label = 'Redshift (z)', format = FuncFormatter(fmt))
plot(m_c15, m_c15,  '--k', linewidth = 0.1)
xlabel(r'$log_{10}(M_{C15}/M_{\odot})$')
ylabel(r'$log_{10}(M_{M-Z}/M_{\odot})$')
tight_layout()
savefig('5param_C15_subsample_5bands_massmet_est_vs_truth.png', dpi=600)
close()

#Plotting (mass estimate - previous mass estimate) over redshift
rcParams["figure.figsize"] = [8,6]
scatter(z_list, m_list-m_c15, s=2.0, c=m_c15)
plot(z_list, np.zeros(len(z_list)), '--k')
colorbar(label = r'$log_{10}(M_{C15}/M_{\odot})$')
xlabel(r'Redshift ($z$)')
ylabel(r'$log_{10}(M_{M-Z}/M_{C15})$')
savefig('5param_C15_5bands_massmet_mdiff_vs_z.png')
close()

#Plotting range from +1sigma to -1sigma over redshift
rcParams["figure.figsize"] = [8,6]
scatter(z_list, m_upper-m_lower, s=2.0, c=m_c15)
plot(z_list, np.zeros(len(z_list)), '--k')
colorbar(label = r'$log_{10}(M_{M-Z}/M_{\odot})$')
xlabel(r'Redshift ($z$)')
ylabel(r'$log_{10}(M_{+1\sigma}/M_{-1\sigma})$')
savefig('5param_C15_5bands_massmet_random_err.png')
close()

#Redshift distribution of sample
rcParams["figure.figsize"] = [7,7]
hist(z_whole, bins = np.linspace(0, max(z_whole), 60), density=1, fc=(0, 0, 1, 0.5), label='Entire COSMOS15 Photometric Sample')
hist(z_list, bins = np.linspace(0, max(z_whole), 60), density=1, fc=(1, 0, 0, 0.5), label='Subsample of {0} galaxies'.format(len(C15)))
legend()
xlabel(r'Redshift ($z$)')
savefig('5param_C15_subsample_5bands_massmet_z_hist.png')
close()

#Plot preliminary version of GSMF within redshift bins as given by Davidzon et al. 2017
z_bins = [0.05, 0.1, 0.2, 0.5, 0.8, 1.1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.5][:-6]

for i in range(len(z_bins)-1):
    logM = m_list[(z_bins[i] <= np.array(z_list)) & (np.array(z_list) < z_bins[i+1])]         #Read stellar masses in Msun
    nbins = 12                              #Number of bins to divide data into
    V     = 1e5                             #Survey volume in Mpc3
    Phi,edg = np.histogram(logM,bins=nbins) #Unnormalized histogram and bin edges
    dM    = edg[1] - edg[0]                 #Bin size
    Max   = edg[0:-1] + dM/2.               #Mass axis
    Phi   = Phi / V / dM                    #Normalize to volume and bin size

    plot(Max, Phi, ls='steps-post', label = r'{0} < z < {1}'.format(z_bins[i], z_bins[i+1]))

plot(0,0, markersize=0)
yscale('log')
xlim([min(m_list), max(m_list)])
xlabel(r'$\log(M_\star\,/\,M_\odot)$')
ylabel(r'$\Phi\,/\,\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}$')
legend(loc='best')
savefig('5param_C15_subsample_5bands_massmet_gsmf.png')
