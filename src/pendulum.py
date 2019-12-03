### Modules ###
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from scipy import stats


sys.path.append('C:/Users/nicol/OneDrive/python_modules')
from ExternalFunctions import nice_string_output, add_text_to_ax, Chi2Regression


### Functions to extract data ###
def read_pendulum_data(filename):
    """
    Read all times in .dat file and append them to a list.
    """
    f = open(filename, 'r')
    times = []

    for line in f:
        
            line = line.split()
            line[0] = float(line[0])
            line[1] = float(line[1])
            times.append(line[1])
    times = np.array(times)
    times = times - times[0]
    return times


### Data ###
# Extracting period data
times_1 = read_pendulum_data('Pendulum/andreas.dat')
times_2 = read_pendulum_data('Pendulum/andrew.dat')
times_3 = read_pendulum_data('Pendulum/Frederik.dat')
times_4 = read_pendulum_data('Pendulum/nicolai.dat')

# Test plot
periods = np.linspace(0, 24, 25)
plt.plot(times_1, periods)
plt.plot(times_2, periods)
plt.plot(times_3, periods)
plt.plot(times_4, periods)
plt.show()

# Measured lengths
pendulum_length = [198.79, 198.81, 198.82, 198.79] # cm 
pendulum_length_after = [198.75, 198.74, 198.675, 198.65] # cm
pendulum_length_laser = [198.7, 198.4, 198.6, 198.2]


### Determine the period of a swing by fitting ###
def fitting_function(x, alpha0, alpha1):
    return alpha0 + alpha1*x

def find_time_errors(x, y):
    # Get slope and intercept
    lin_regress = stats.linregress(x, y)

    residual = []
    for i in range(25):
        fitted_time = fitting_function(x[i], lin_regress[1], lin_regress[0])
        residual.append(y[i] - fitted_time)

    N_bins = 50
    plt.hist(residual, bins=N_bins, range=[-0.2,0.2], histtype='step')
    

find_time_errors(periods, times_1)



def chi2_pendulum_fit(x, y):
    chi2_object = Chi2Regression(fitting_function, x, y) 
    minuit = Minuit(chi2_object, pedantic=False, alpha0=0.0, alpha1=0.4, print_level=0)  
    minuit.migrad() # Perform the actual fit
    minuit_output = [minuit.get_fmin(), minuit.get_param_states()] # Save the output parameters in case needed
    
    # Here we extract the fitting parameters and their errors
    alpha0_fit = minuit.values['alpha0']
    alpha1_fit = minuit.values['alpha1']
    sigma_alpha0_fit = minuit.errors['alpha0']
    sigma_alpha1_fit = minuit.errors['alpha1']

    Npoints = 25 # Number of data points
    Nvar = 2 # Number of variables
    Ndof_fit = Npoints - Nvar # Number of degrees of freedom

    Chi2_fit = minuit.fval # The chi2 value
    Prob_fit = stats.chi2.sf(Chi2_fit, Ndof_fit) # The chi2 probability given N degrees of freedom

    # Plotting
    fig, ax = plt.subplots(figsize=(10,6))
    plotting_times = np.linspace(0.0, 26, 1000)
    ax.errorbar(x, y, fmt='ro', ecolor='k', elinewidth=1, capsize=2, capthick=1)
    ax.plot(plotting_times, fitting_function(plotting_times, alpha0_fit, alpha1_fit), '-r')
    # Add nice text
    d = {'alpha0':  [alpha0_fit, sigma_alpha0_fit],
        'alpha1':   [alpha1_fit, sigma_alpha1_fit],
        'Chi2':     Chi2_fit,
        'ndf':      Ndof_fit,
        'Prob':     Prob_fit,
        }
    text = nice_string_output(d, extra_spacing=2, decimals=3)
    add_text_to_ax(0.02, 0.95, text, ax, fontsize=14)
    fig.tight_layout()
    plt.show()


chi2_pendulum_fit(periods, times_1)












def  pendulum_g(lengt, period, err_length, err_period):
    g = lengt * ((2*np.pi)/period)**2
    err_g = np.sqrt((2*np.pi/period)**4 * err_length**2 + (8*np.pi**2*lengt / period**3)**2 * err_period**2)
    return g, err_g