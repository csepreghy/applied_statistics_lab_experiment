# Analyze pendulum data
# Use this notebook to quickly test whether your pendulum data makes sense!

import numpy as np                                     # Matlab like syntax for linear algebra and functions
import matplotlib.pyplot as plt                        # Plots and figures like you know them from Matlab
import seaborn as sns                                  # Make the plots nicer to look at
from iminuit import Minuit                             # The actual fitting tool, better than scipy's
import sys                                             # Module to see files and folders in directories
from scipy import stats
from ExternalFunctions import Chi2Regression
from ExternalFunctions import nice_string_output, add_text_to_ax # useful functions to print fit results on figure

# Example file location
filename1 = 'data/Pendulum/andrew.dat'
filename2 = 'data/Pendulum/nicolai.dat'
filename3 = 'data/Pendulum/andreas.dat'
filename4 = 'data/Pendulum/Frederik.dat'


# Read in data
def read_data(filename):
    dat = np.genfromtxt(filename, delimiter='\t', names=('n', 't_s'))
    return dat


# Read and plot the data
data_example = read_data(filename1)
n, t = data_example['n'], data_example['t_s']

# Plotting
sig_t = 0.1     # Set your own values...
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 10),
                       gridspec_kw={'height_ratios':[4,1]}, sharex=True)
ax[0].errorbar(n, t, yerr=sig_t, color='k', fmt='o')
# ax[0].plot(n_plot, t_plot3)
ax[0].set_xlabel('Timing measurement number')
ax[0].set_ylabel('Time elapsed (s)')
ax[0].set(xlim=(0, n[-1]+np.ediff1d(n)[0]), ylim=(0, t[-1]+np.ediff1d(t)[0]))

plt.show()

data_1 = read_data(filename1)
data_2 = read_data(filename2)
data_3 = read_data(filename3)
data_4 = read_data(filename4)
n1, t1 = data_1['n'], data_1['t_s']
n2, t2 = data_2['n'], data_2['t_s']
n3, t3 = data_3['n'], data_3['t_s']
n4, t4 = data_4['n'], data_4['t_s']
print(t1, t2, t3, t4)

# Plotting raw data
sig_t = 0.05     # Set your own values...
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(30, 5), sharex=True)
ax[0].errorbar(n1, t1, yerr=sig_t, color='k', fmt='o')
ax[0].set_xlabel('Timing measurement number')
ax[0].set_ylabel('Time elapsed (s)')
ax[0].set(xlim=(0, n1[-1]+np.ediff1d(n)[0]), ylim=(0, t1[-1]+np.ediff1d(t)[0]))
ax[0].set_title('Experiment 1')
ax[1].errorbar(n2, t2, yerr=sig_t, color='k', fmt='o')
ax[1].set_xlabel('Timing measurement number')
ax[1].set_ylabel('Time elapsed (s)')
ax[1].set(xlim=(0, n2[-1]+np.ediff1d(n)[0]), ylim=(0, t2[-1]+np.ediff1d(t)[0]))
ax[1].set_title('Experiment 2')
ax[2].errorbar(n3, t3, yerr=sig_t, color='k', fmt='o')
ax[2].set_xlabel('Timing measurement number')
ax[2].set_ylabel('Time elapsed (s)')
ax[2].set(xlim=(0, n3[-1]+np.ediff1d(n)[0]), ylim=(0, t3[-1]+np.ediff1d(t)[0]))
ax[2].set_title('Experiment 3')
ax[3].errorbar(n4, t4, yerr=sig_t, color='k', fmt='o')
ax[3].set_xlabel('Timing measurement number')
ax[3].set_ylabel('Time elapsed (s)')
ax[3].set(xlim=(0, n4[-1]+np.ediff1d(n)[0]), ylim=(0, t4[-1]+np.ediff1d(t)[0]))
ax[3].set_title('Experiment 4')

plt.show()

def fit_function(x, a, b):
    return a * x + b

def linfit(x, y, a0, b0, e):
    if e > 0:
        chi2_object = Chi2Regression(fit_function, x, y, e)
    else:
        chi2_object = Chi2Regression(fit_function, x, y)
    minuit = Minuit(chi2_object, pedantic=False, b=b0, a=a0, print_level=0)
    minuit.migrad();
    minuit_output = [minuit.get_fmin(), minuit.get_param_states()]
    b_fit = minuit.values['b']
    a_fit = minuit.values['a']
    sigma_b_fit = minuit.errors['b']
    sigma_a_fit = minuit.errors['a']
    Nvar = 2
    Ndof_fit = len(x) - Nvar    
    Chi2_fit = minuit.fval                          # The chi2 value
    Prob_fit = stats.chi2.sf(Chi2_fit, Ndof_fit)    # The chi2 probability given N degrees of freedom
    return a_fit, b_fit, sigma_a_fit, sigma_b_fit, Chi2_fit, Prob_fit

error = 0.0
fit1 = linfit(n1, t1, 2.5, 0.0, error)
fit2 = linfit(n2, t2, 2.5, 0.0, error)
fit3 = linfit(n3, t3, 2.5, 0.0, error)
fit4 = linfit(n4, t4, 2.5, 0.0, error)

print(fit1)
print(fit2)
print(fit3)
print(fit4)

def res(n, t, fit):
    t1_fit = fit[0] * n + fit[1]
    residual = t1_fit - t
    print(f"The fit gave: T = {fit[0]:.4f} +- {fit[2]:.10f} seconds")
    return t1_fit, residual

t1_res = res(n1, t1, fit1)[1]
t2_res = res(n2, t2, fit2)[1]
t3_res = res(n3, t3, fit3)[1]
t4_res = res(n4, t4, fit4)[1]


# Plotting raw data
sig_t = 0.05     # Set your own values...
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(28, 10), sharex=False)
ax[0,0].errorbar(n1, t1, yerr=sig_t, color='k', fmt='o')
ax[0,0].set_xlabel('Timing measurement number')
ax[0,0].set_ylabel('Time elapsed (s)')
ax[0,0].set(xlim=(0, n1[-1]+np.ediff1d(n)[0]), ylim=(0, t1[-1]+np.ediff1d(t)[0]))
ax[0,0].set_title('Experiment 1')
ax[1,0].hist(t1_res, bins=50,range=(-0.2, 0.2), histtype='step')
ax[1,0].set_title('Histogram of t1_res')
string = " Entries {:>6} \n Mean {:>9.3f} \n RMS {:>10.3f}".format(len(t1_res), t1_res.mean(), t1_res.std(ddof=1))
ax[1,0].text(0.62, 0.95, string, family='monospace', transform=ax[1,0].transAxes, fontsize=12, verticalalignment='top')
ax[0,1].errorbar(n2, t2, yerr=sig_t, color='k', fmt='o')
ax[0,1].set_xlabel('Timing measurement number')
ax[0,1].set_ylabel('Time elapsed (s)')
ax[0,1].set(xlim=(0, n2[-1]+np.ediff1d(n)[0]), ylim=(0, t2[-1]+np.ediff1d(t)[0]))
ax[0,1].set_title('Experiment 2')
ax[1,1].hist(t2_res, bins=50, range=(-0.2, 0.2), histtype='step')
ax[1,1].set_title('Histogram of t2_res')
string = " Entries {:>6} \n Mean {:>9.3f} \n RMS {:>10.3f}".format(len(t2_res), t2_res.mean(), t2_res.std(ddof=1))
ax[1,1].text(0.62, 0.95, string, family='monospace', transform=ax[1,1].transAxes, fontsize=12, verticalalignment='top')
ax[0,2].errorbar(n3, t3, yerr=sig_t, color='k', fmt='o')
ax[0,2].set_xlabel('Timing measurement number')
ax[0,2].set_ylabel('Time elapsed (s)')
ax[0,2].set(xlim=(0, n3[-1]+np.ediff1d(n)[0]), ylim=(0, t3[-1]+np.ediff1d(t)[0]))
ax[0,2].set_title('Experiment 3')
ax[1,2].hist(t3_res, bins=50, range=(-0.2, 0.2), histtype='step')
ax[1,2].set_title('Histogram of t3_res')
string = " Entries {:>6} \n Mean {:>9.3f} \n RMS {:>10.3f}".format(len(t3_res), t3_res.mean(), t3_res.std(ddof=1))
ax[1,2].text(0.62, 0.95, string, family='monospace', transform=ax[1,2].transAxes, fontsize=12, verticalalignment='top')
ax[0,3].errorbar(n4, t4, yerr=sig_t, color='k', fmt='o')
ax[0,3].set_xlabel('Timing measurement number')
ax[0,3].set_ylabel('Time elapsed (s)')
ax[0,3].set(xlim=(0, n4[-1]+np.ediff1d(n)[0]), ylim=(0, t4[-1]+np.ediff1d(t)[0]))
ax[0,3].set_title('Experiment 4')
ax[1,3].hist(t4_res, bins=50, range=(-0.2, 0.2), histtype='step')
ax[1,3].set_title('Histogram of t4_res')
string = " Entries {:>6} \n Mean {:>9.3f} \n RMS {:>10.3f}".format(len(t4_res), t4_res.mean(), t4_res.std(ddof=1))
ax[1,3].text(0.62, 0.95, string, family='monospace', transform=ax[1,3].transAxes, fontsize=12, verticalalignment='top')

res_all = np.concatenate((t1_res, t2_res, t3_res, t4_res))
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), sharex=False)
ax.hist(res_all, bins=100, range=(-0.15, 0.15), histtype='step')
ax.set_title('Histogram of residuals of all expiriments')
string = " Entries {:>6} \n Mean {:>9.3f} \n RMS {:>10.3f}".format(len(res_all), res_all.mean(), res_all.std(ddof=1))
ax.text(0.62, 0.95, string, family='monospace', transform=ax.transAxes, fontsize=12, verticalalignment='top')

plt.tight_layout()
plt.show()

#Refit with RMS of residuals as error
RMS = res_all.std(ddof=1)
fit1 = linfit(n1, t1, fit1[0], fit1[1], t1_res.std(ddof=1))
fit2 = linfit(n2, t2, fit2[0], fit2[1], t2_res.std(ddof=1))
fit3 = linfit(n3, t3, fit3[0], fit3[1], t3_res.std(ddof=1))
fit4 = linfit(n4, t4, fit4[0], fit4[1], t4_res.std(ddof=1))
print(f"The fit of data set 1: T = {fit1[0]:.6f} +- {fit1[2]:.6f} seconds")
print(f"The fit of data set 2: T = {fit2[0]:.6f} +- {fit2[2]:.6f} seconds")
print(f"The fit of data set 3: T = {fit3[0]:.6f} +- {fit3[2]:.6f} seconds")
print(f"The fit of data set 4: T = {fit4[0]:.6f} +- {fit4[2]:.6f} seconds")

# # What to do next?
# 
# The reason for the (empty) plot below is, that though your data (hopefully) lies on a line, you're not able to see any smaller effects. However, if you plot the __residuals__ (the difference between your measurements and a fit to them), then you will much better be able to tell, if the data looks good. Also, it is from a historgram of the residuals, that you can determine, if your errors are Gaussian, and from the RMS what your (typical) uncert
