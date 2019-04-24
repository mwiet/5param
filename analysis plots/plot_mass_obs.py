import pandas as pd
import numpy as np
from matplotlib.pyplot import *
from scipy.odr import *
import matplotlib
from matplotlib.ticker import FuncFormatter 
from astropy.cosmology import WMAP9 as cosmo
from scipy.integrate import quad
from pandas.compat import StringIO

switch_backend('agg')
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

C15_2000 = pd.read_csv('c15_clean_2000.csv')
C15_264 = pd.read_csv('c15_clean_264.csv')
C15_6000 = pd.read_csv('c15_clean_6000.csv')
C15 = pd.concat([C15_2000, C15_264, C15_6000], sort=False)
z_whole = pd.read_csv('C15_clean.csv')['ZPDF']
m_list = []
z_list = []
m_upper = []
m_lower = []
remove_list = []
for index, row in C15.iterrows():
    i = int(row['NUMBER'])
    try:
        params = pd.read_csv('_5bands_emcee_5param_obs_' + str(i) + '_estimates.csv') #Change names according to outroot defined in 'prospector_[...]_run.py' file (currently only plotting params)
        #params = pd.read_csv('emcee_5param_obs_' + str(i) + '_estimates.csv') 
        #params3 = pd.read_csv('_5bands_emcee_5param_obs_' + str(i) + '_estimates.csv')
        m_list.append(float(params['mass']))
        m_upper.append(float(params['mass_upper']))
        m_lower.append(float(params['mass_lower']))
        z_list.append(float(params['zred']))
    except:
        #print('Did not find ' + str(i))
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
    return x + B[0]

linear = Model(f)

m_c15_err_avg = abs(np.array(m_c15_err)).mean(axis=0)
m_err_avg = abs(np.array(m_err)).mean(axis=0)

m_c15f =  m_c15[~np.isnan(np.array(m_c15_err_avg))]
m_listf =  m_list[~np.isnan(np.array(m_c15_err_avg))]
m_err_avgf =  m_err_avg[~np.isnan(np.array(m_c15_err_avg))]
m_c15_err_avgf =  m_c15_err_avg[~np.isnan(np.array(m_c15_err_avg))]
print(len(m_c15f[m_c15f<10]))
mydata = RealData(m_c15f[m_c15f<10], m_listf[m_c15f<10], sx=m_c15_err_avgf[m_c15f<10], sy=m_err_avgf[m_c15f<10])
myodr = ODR(mydata, linear, beta0=[1., 0.])
myoutput = myodr.run()
myoutput.pprint()

#Plotting mass estimate vs previous estimate
rcParams["figure.figsize"] = [8,7]
cmap = matplotlib.cm.get_cmap('jet')
norm = matplotlib.colors.Normalize(vmin=min(z_list), vmax=max(z_list))

scatter(m_c15, m_list, c=z_list, cmap=cmap, s=0.1)
errorbar(m_c15, m_list, xerr = m_c15_err, yerr = m_err, fmt = 'o', markersize=0, linewidth=0.03, capsize=3.0, capthick = 0.03, c='grey')
scatter(m_c15, m_list, c=z_list, cmap=cmap, s=0.1)

def fmt(x, pos):
    z = 10**(x)
    return r'${}$'.format(round(z,3))

colorbar(label = 'Redshift (z)')#, format = FuncFormatter(fmt))
plot(m_c15, m_c15,  '--k', linewidth = 0.1)
xlabel(r'$log_{10}(M_{C15}/M_{\odot})$')
ylabel(r'$log_{10}(M_{est}/M_{\odot})$')
tight_layout()
savefig('5param_C15_subsample_8000_5bands_t_est_vs_truth.png', dpi=600)
close()

#Plotting (mass estimate - previous mass estimate) over redshift
rcParams["figure.figsize"] = [8,6]
scatter(z_list, m_list-m_c15, s=2.0, c=m_c15)
plot(z_list, np.zeros(len(z_list)), '--k')
colorbar(label = r'$log_{10}(M_{C15}/M_{\odot})$')
xlabel(r'Redshift ($z$)')
ylabel(r'$log_{10}(M_{est}/M_{C15})$')
savefig('5param_C15_8000_5bands_t_mdiff_vs_z.png')
close()

#Plotting range from +1sigma to -1sigma over redshift
rcParams["figure.figsize"] = [8,6]
cmap = matplotlib.cm.get_cmap('plasma')
scatter(z_list, m_upper-m_lower, s=2.0, c=m_list, cmap = cmap)
#plot(z_list, np.zeros(len(z_list)), '--k')
colorbar(label = r'$log_{10}(M_{est}/M_{\odot})$')
xlabel(r'Redshift ($z$)')
ylabel(r'$log_{10}(M_{+1\sigma}/M_{-1\sigma})$')
savefig('5param_C15_8000_5bands_random_err.png')
close()


#Redshift distribution of sample
rcParams["figure.figsize"] = [7,7]
hist(z_whole, bins = np.linspace(0, max(z_whole), 60), density=1, fc=(0, 0, 1, 0.5), label='Entire COSMOS15 Photometric Sample')
hist(z_list, bins = np.linspace(0, max(z_whole), 60), density=1, fc=(1, 0, 0, 0.5), label='Subsample of {0} galaxies'.format(len(C15)))
legend()
xlabel(r'Redshift ($z$)')
savefig('5param_C15_subsample_8000_5bands_z_hist.png')
close()

#Plot preliminary version of GSMF within redshift bins as given by Davidzon et al. 2017
z_bins = [0.05, 0.1, 0.2, 0.5, 0.8, 1.1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.5][:-8][2:]

def dVdz(z):
    return cosmo.differential_comoving_volume(z).value

def cosmic_stddev(z_lower, z_upper, logM_bins):
    #Cosmic Variance as given by Moster et al., 2011 for COSMOS survey using WMAP3 cosmology
    z_mean = (z_upper+z_lower)/2
    z_diff = abs(z_upper - z_lower)
    sigma_dm = 0.069/(0.234+z_mean**(0.824)) #standard dev of variation induced by fluctuations in the DM distribution
    sigma_gg_list = []
    for logM in logM_bins:
        if logM <=9.0:
            b0 = 0.062
            b1 = 2.59
            b2 = 1.025
        elif 9.0 < logM <=9.5:
            b0 = 0.074
            b1 = 2.58
            b2 = 1.039
        elif 9.5 < logM <=10.0:
            b0 = 0.042
            b1 = 3.17
            b2 = 1.147
        elif 10.0 < logM <=10.5:
            b0 = 0.053
            b1 = 3.07
            b2 = 1.225
        elif 10.5 < logM <=11.0:
            b0 = 0.069
            b1 = 3.19
            b2 = 1.269
        elif 11.5 < logM:
            b0 = 0.173
            b1 = 2.89
            b2 = 1.438
        b = (b0*(z_mean+1)**b1) + b2 #proportionality between galaxy correlation function and DM correlation function
        sigma_gg_list.append(b*sigma_dm*np.sqrt(0.2/z_diff)) #standard dev of variation induced by fluctuations in the galaxy distribution
    return np.array(sigma_gg_list)

avg_relsigma = (abs(m_upper-m_lower)/2)/m_list

def SEDrelerr(z_lower, z_upper, logM_array, logM_bins):
    #relative error from SED fitting
    avg_relsigmaz = avg_relsigma[(z_lower <= np.array(z_list)) & (np.array(z_list) < z_upper)]
    relerr = []
    for i in range(len(logM_bins)-1): #over bins from bin edges
        avg_relsigmaM = avg_relsigmaz[(logM_bins[i] <= logM_array) & (logM_array < logM_bins[i+1])]
        relerr.append(avg_relsigmaM.sum()/len(avg_relsigmaM))
    return np.array(relerr)
        
        
    
def doubleschechter(B, x):
    """
    x - stellar mass values of bins in log_{10}(M/Msolar)
    B[0] - Characteristic mass in solar masses
    B[1] - Characteristic density of the first mode
    B[2] - Gradient of the first mode
    B[3] - Characteristic density of the second mode
    B[4] - Gradient of the scond mode
    """
    y = (10**x)/B[0]
    return (1/B[0])*(B[1]*(y**B[2]) + B[3]*(y**B[4]))*np.e**(-y)



def read_data(kind, method, zbin):
    dat = 'mf_mass2b_fl5b_' + str(kind) + '_' + str(method) + str(zbin) + '.dat'
    #print(str(dat))
    df = pd.read_csv(dat, sep="\s+", header = None, names=['logM', 'logSMF', 'logSMFerror', 'logMerror'])
    df['kind'] = str(kind)
    df['method'] = str(method)
    df['zbin'] = zbin
    return df

main_df = pd.DataFrame(columns=[ 'logM', 'logSMF','logSMFerror', 'logMerror', 'kind', 'method', 'zbin'])
for kind in ['tot', 'act', 'pas']:
	for method in ['Vmax', 'VmaxFit2D']:
		for zbin in range(0,9):
			df = read_data(kind, method, zbin) #LOAD Davidzon et al. 2017 data
			main_df = main_df.append(df)
main_df_copy = main_df


def plot_GSMF(main_df, sel_kind, sel_method, sel_zbin, label):
    main_df = main_df[(main_df['kind'] == sel_kind) & (main_df['method'] == sel_method) & (main_df['zbin'] == sel_zbin)]
    if sel_method == 'Vmax':
        errorbar(main_df['logM'],
                     10**np.array(main_df['logSMF']),
                     #xerr = main_df['logMerror'],
                     yerr = (10**np.array(main_df['logSMF']))*(np.array(main_df['logSMFerror'])/np.array(main_df['logSMF'])),
                     marker = '.', label = label)
    else:
        errorbar(main_df['logM'],
                     10**np.array(main_df['logSMF']),
                     #yerr = main_df['logSMFerror'] - main_df['logMerror'],
                     marker = '.', label = label)



for i in range(len(z_bins)-1):
    logM = m_list[(z_bins[i] <= np.array(z_list)) & (np.array(z_list) < z_bins[i+1])]         #Read stellar masses in Msun
    logM_large = logM[logM > np.log10(6.3e7*(1+z_bins[i])**2.7)] #from limiting magnitude of COSMOS survey (Pozzetti et al. 2010)
    nbins = 15 #Number of bins to divide data into
    print('Number of galaxies:')
    print(len(logM_large))
    V   = 0.68*((np.pi/180)**2)*quad(dVdz, z_bins[i], z_bins[i+1])[0] #Survey volume in Mpc3 (survey solid angle assumed to be minimum from UD COSMOS Field)
    Phi,edg = np.histogram(logM_large,bins=nbins) #Unnormalized histogram and bin edges
    dM    = edg[1] - edg[0]                 #Bin size
    Max   = edg[0:-1] + dM/2.               #Mass axis
    Phi_err = np.sqrt(Phi)/V # + (SEDrelerr(z_bins[i], z_bins[i+1], logM, edg)*Phi)**2) #Poisson error + cosmic variance + SED fitting error
    Phi   = Phi / V / dM
    

    #plot(Max, Phi, ls='-', marker='.', label = r'{0} < z < {1}'.format(z_bins[i], z_bins[i+1]))
    Phi_err = np.sqrt(Phi_err**2 + (SEDrelerr(z_bins[i], z_bins[i+1], logM, edg)*Phi)**2 +(cosmic_stddev(z_bins[i], z_bins[i+1], Max)*Phi)**2)
    errorbar(Max, Phi, ls='-', marker='.', label = r'{0} < z < {1}'.format(z_bins[i], z_bins[i+1]), yerr=Phi_err)
    plot_GSMF(main_df_copy, 'tot', 'Vmax', i,  r'{0} < z < {1} (Davidzon et al., 2017)'.format(z_bins[i], z_bins[i+1]))
    print(r'{0} < z < {1}'.format(z_bins[i], z_bins[i+1]))
    mydata = RealData(Max, Phi, sy=Phi_err)
    myodr = ODR(mydata, Model(doubleschechter), beta0=[1e11, 1.2e-3, -1.4, 1.9e-3, -0.4])
    myoutput = myodr.run()
    myoutput.pprint()
    #plot(np.linspace(8.5, 12, 30), doubleschechter(myoutput.beta, np.linspace(8.5, 12, 30)))

plot(0,0, markersize=0)
yscale('log')
xlim([9,12])
xlabel(r'$log_{10}(M/\,M_\odot)$')
ylabel(r'$\Phi\,/\,\mathrm{dex}^{-1}\,\mathrm{Mpc}^{-3}$')
legend(loc='best')
savefig('5param_C15_subsample_8000_5bands_gsmf.png')

