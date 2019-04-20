import pandas as pd
import numpy as np
from matplotlib.pyplot import *
import matplotlib
from scipy.odr import *

switch_backend('agg')
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


m_true_list, m_list, z_list, m_upper, m_lower, remove_list = [], [], [], [], [], []
m_ind = np.linspace(6.5, 11.5, 6)
z_ind = np.logspace(np.log10(0.005), np.log10(5.8), 10, endpoint=True).tolist() #+ np.logspace(np.log10(5.8), np.log10(10), 5, endpoint=True).tolist()[1:]
z_ind = [round(z, 6) for z in z_ind]
for z in z_ind:
    for m in m_ind:
        try:
            params = pd.read_csv('emcee_5param_mock_5bands_age_m=' + str(float(m)) + '_z=' + str(float(z)) + '_estimates.csv') #Change names according to outroot defined in 'prospector_[...]_run.py' file 
            m_true_list.append(float(params['mass_true']))
            z_list.append(float(params['zred']))
            m_list.append(float(params['mass']))
            m_upper.append(float(params['mass_upper']))
            m_lower.append(float(params['mass_lower']))
        except:
            print('m=' + str(m) + ' & z=' +str(z) + ' not in directory')

m_list = np.log10(np.array(m_list))
m_upper = np.log10(np.array(m_upper))
m_lower = np.log10(np.array(m_lower))
m_true_list = np.log10(np.array(m_true_list))
m_err = [m_list-m_lower, m_upper-m_list]

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
mydata = RealData(m_true_list, m_list, sy=np.array(m_err).mean(axis=0))
myodr = ODR(mydata, linear, beta0=[1., 0.])
myoutput = myodr.run()
myoutput.pprint()

#Plotting mass estimate vs truth
cmap = matplotlib.cm.get_cmap('jet')
norm = matplotlib.colors.Normalize(vmin=min(z_list), vmax=max(z_list))

rcParams["figure.figsize"] = [8,7]
errorbar(m_true_list, m_list, yerr = m_err, fmt = 'o', markersize=0, linewidth=0.1, capsize=4.0, capthick = 0.1, c='grey')
scatter(m_true_list, m_list, c=np.log10(z_list), cmap=cmap, s=2.0)

from matplotlib.ticker import FuncFormatter 

def fmt(x, pos):
    z = 10**(x)
    return r'${}$'.format(round(z,3))

colorbar(ticks=np.log10(z_list), label = 'Redshift (z)', format = FuncFormatter(fmt))
plot(m_true_list, m_true_list, '--k', linewidth = 0.1)
xlabel(r'$log_{10}(M_{truth}/M_{\odot})$')
ylabel(r'$log_{10}(M_{est}/M_{\odot})$')
tight_layout()
savefig('5param_mock_5bands_est_vs_truth.png', dpi=600)
close()

#Plotting (mass estimate - truth) over redshift
rcParams["figure.figsize"] = [8,6]
scatter(z_list, m_list-m_true_list, s=2.0, c=m_true_list)
colorbar(label = r'$log_{10}(M_{truth}/M_{\odot})$')
plot(z_list, np.zeros(len(z_list)), '--k', linewidth = 0.1)
xlabel(r'Redshift ($z$)')
ylabel(r'$log_{10}(M_{est}/M_{truth})$')
savefig('5param_mock_5bands_mdiff_vs_z.png')
close()

#Plotting range from +1sigma to -1sigma over redshift
rcParams["figure.figsize"] = [8,6]
scatter(z_list, m_upper-m_lower, s=2.0, c=m_true_list)
colorbar(label = r'$log_{10}(M_{truth}/M_{\odot})$')
xlabel(r'Redshift ($z$)')
ylabel(r'$log_{10}(M_{+1\sigma}/M_{-1\sigma})$')
savefig('5param_mock_5bands_random_err.png')
close()

#Redshift distribution of sample
rcParams["figure.figsize"] = [7,7]
hist(z_list, bins = 10 ** np.linspace(np.log10(min(z_ind)), np.log10(max(z_ind)+1), 20))
xlabel(r'Redshift ($z$)')
savefig('5param_mock_5bands_z_hist.png')
close()
