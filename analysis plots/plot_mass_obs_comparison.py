import pandas as pd
import numpy as np
from matplotlib.pyplot import *
from matplotlib.ticker import FuncFormatter 
import matplotlib
from scipy.odr import *

switch_backend('agg')
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

C15 = pd.read_csv('c15_clean_264.csv')
z_whole = pd.read_csv('C15_clean.csv')['ZPDF']
z_list = []
m_list_1 = []
m_upper_1 = []
m_lower_1 = []
m_list_2 = []
m_upper_2 = []
m_lower_2 = []
remove_list = []
for index, row in C15.iterrows():
    i = int(row['NUMBER'])
    try:
        #Change names according to outroot defined in 'prospector_[...]_run.py' file
        #Currently comparing these following two models:
        params = pd.read_csv('_5bands_emcee_5param_obs_' + str(i) + '_estimates.csv') 
        params2 = pd.read_csv('_5bands_emcee_5param_obs_massmet_' + str(i) + '_estimates.csv')
        m_list_1.append(float(params['mass']))
        m_upper_1.append(float(params['mass_upper']))
        m_lower_1.append(float(params['mass_lower']))
        m_list_2.append(float(params2['mass']))
        m_upper_2.append(float(params2['mass_upper']))
        m_lower_2.append(float(params2['mass_lower']))
        z_list.append(float(params['zred']))
    except:
        print('Did not find ' + str(i))
        remove_list.append(i)
    
    
       
m_list_1 = np.log10(np.array(m_list_1))
m_upper_1 = np.log10(np.array(m_upper_1))
m_lower_1 = np.log10(np.array(m_lower_1))
m_err_1 = [m_list_1-m_lower_1, m_upper_1-m_list_1]

m_list_2 = np.log10(np.array(m_list_2))
m_upper_2 = np.log10(np.array(m_upper_2))
m_lower_2 = np.log10(np.array(m_lower_2))
m_err_2 = [m_list_2-m_lower_2, m_upper_2-m_list_2]

for i in remove_list:
    C15 = C15[C15['NUMBER'] != i]

print('Plotting ' + str(len(C15)) + ' galaxies')

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
mydata = RealData(m_list_1, m_list_2, sx=np.array(m_err_1).mean(axis=0), sy=np.array(m_err_2).mean(axis=0))
myodr = ODR(mydata, linear, beta0=[1., 0.])
myoutput = myodr.run()
myoutput.pprint()

#Plotting mass estimate vs other estimate
rcParams["figure.figsize"] = [8,7]
cmap = matplotlib.cm.get_cmap('jet')
norm = matplotlib.colors.Normalize(vmin=min(z_list), vmax=max(z_list))

errorbar(m_list_1, m_list_2, xerr = m_err_1, yerr = m_err_2, fmt = 'o', markersize=0, linewidth=0.1, capsize=4.0, capthick = 0.1, c='grey')
scatter(m_list_1, m_list_2, c=np.log10(z_list), cmap=cmap, s=2.0)

def fmt(x, pos):
    z = 10**(x)
    return r'${}$'.format(round(z,3))

colorbar(label = 'Redshift (z)', format = FuncFormatter(fmt))
plot(m_list_1, m_list_1,  '--k', linewidth = 0.1)
xlabel(r'$log_{10}(M_{est}/M_{\odot})$')
ylabel(r'$log_{10}(M_{M-Z}/M_{\odot})$')
tight_layout()
savefig('5param_comparison_subsample_massmet_est_vs_truth.png', dpi=600)
close()

#Plotting (mass estimate - other mass estimate) over redshift
rcParams["figure.figsize"] = [8,6]
scatter(z_list, m_list_2-m_list_1, s=2.0, c=m_list_1)
plot(z_list, np.zeros(len(z_list)), '--k')
colorbar(label = r'$log_{10}(M_{est}/M_{\odot})$')
xlabel(r'Redshift ($z$)')
ylabel(r'$log_{10}(M_{M-Z}/M_{est})$')
savefig('5param_comparison_massmet_mdiff_vs_z.png')
close()

#Redshift distribution
rcParams["figure.figsize"] = [7,7]
hist(z_whole, bins = np.linspace(0, max(z_whole), 60), density=1, fc=(0, 0, 1, 0.5), label='Entire COSMOS15 Photometric Sample')
hist(z_list, bins = np.linspace(0, max(z_whole), 60), density=1, fc=(1, 0, 0, 0.5), label='Subsample of {0} galaxies'.format(len(C15)))
legend()
xlabel(r'Redshift ($z$)')
savefig('5param_comparison_massmet_subsample_z_hist.png')
close()

