### Modules ###
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from scipy import stats

from ExternalFunctions import nice_string_output, add_text_to_ax, Chi2Regression

### Functions for extracting data ###
def read_ball_data(filename):
    """
    Read all time and voltage data points in .csv file and append them to two lists.
    """
    f = open(filename, 'r')
    time = []
    voltage = []

    for i, line in enumerate(f):
        if i >= 15:
            line = line.split(',')
            line[0] = float(line[0])
            line[1] = float(line[1])
            time.append(line[0])
            voltage.append(line[1])
    return time, voltage

def time_at_gate(time, voltage):
    """
    For a single experiment, defines the time where the ball passes each og the 5 gates and assign uncertaincy.
    """
    gate_t = []
    for i in range(len(time)):
        if np.sign(voltage[i]) != np.sign(voltage[i-1]):
            gate_t.append(time[i])
    
    t_gates = []
    err_t_gates = []
    for i in range(5):
        t_gates.append((gate_t[2*i+1]+gate_t[2*i])/2 - (gate_t[1]+gate_t[0])/2)
        err_t_gates.append((gate_t[2*i+1]-gate_t[2*i])/2)
    #print(t_gates)
    return t_gates, err_t_gates

def collect_gate_times(directory):
    """
    Collects all the gate times from different experiments and finds the central value through a weighted average.
    The errors of each experiment is propergated.
    """
    # Lists containing the gate times and their errors
    t_gate_1 = []
    t_gate_2 = []
    t_gate_3 = []
    t_gate_4 = []
    t_gate_5 = []
    err_t_gate_1 = []
    err_t_gate_2 = []
    err_t_gate_3 = []
    err_t_gate_4 = []
    err_t_gate_5 = []

    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        time, voltage = read_ball_data(path)
        t_gates, err_t_gates = time_at_gate(time, voltage)
        # Append gate times and errors
        t_gate_1.append(t_gates[0])
        t_gate_2.append(t_gates[1])
        t_gate_3.append(t_gates[2])
        t_gate_4.append(t_gates[3])
        t_gate_5.append(t_gates[4])
        err_t_gate_1.append(err_t_gates[0])
        err_t_gate_2.append(err_t_gates[1])
        err_t_gate_3.append(err_t_gates[2])
        err_t_gate_4.append(err_t_gates[3])
        err_t_gate_5.append(err_t_gates[4])

    err_t_gate_1 = np.array(err_t_gate_1)
    err_t_gate_2 = np.array(err_t_gate_2)
    err_t_gate_3 = np.array(err_t_gate_3)
    err_t_gate_4 = np.array(err_t_gate_4)
    err_t_gate_5 = np.array(err_t_gate_5)
    
    weighted_average_t_gate_1 = np.sum(t_gate_1/err_t_gate_1**2) / np.sum(1/err_t_gate_1**2)
    weighted_average_t_gate_2 = np.sum(t_gate_2/err_t_gate_2**2) / np.sum(1/err_t_gate_2**2)
    weighted_average_t_gate_3 = np.sum(t_gate_3/err_t_gate_3**2) / np.sum(1/err_t_gate_3**2)
    weighted_average_t_gate_4 = np.sum(t_gate_4/err_t_gate_4**2) / np.sum(1/err_t_gate_4**2)
    weighted_average_t_gate_5 = np.sum(t_gate_5/err_t_gate_5**2) / np.sum(1/err_t_gate_5**2)
    gate_times = np.array([weighted_average_t_gate_1, weighted_average_t_gate_2, weighted_average_t_gate_3, weighted_average_t_gate_4, weighted_average_t_gate_5])

    prop_err_t_gate_1 = np.sqrt(np.sum(err_t_gate_1)**2)
    prop_err_t_gate_2 = np.sqrt(np.sum(err_t_gate_2)**2)
    prop_err_t_gate_3 = np.sqrt(np.sum(err_t_gate_3)**2)
    prop_err_t_gate_4 = np.sqrt(np.sum(err_t_gate_4)**2)
    prop_err_t_gate_5 = np.sqrt(np.sum(err_t_gate_5)**2)
    err_gate_times = np.array([prop_err_t_gate_1, prop_err_t_gate_2, prop_err_t_gate_3, prop_err_t_gate_4, prop_err_t_gate_5])
    print()
    print("Weighted Average times")
    print(gate_times)
    print()
    return gate_times, err_gate_times


### Data ###
# Importing date from the experiments
gate_times, err_gate_times = collect_gate_times('./data/Measurements_1')
reverse_gate_times, reverse_err_gate_times = collect_gate_times('./data/Measurements_2')

# Path measurements in cm
gate_1 = np.array([22.28, 22.45, 22.52, 22.42])
gate_2 = np.array([35.36, 35.62, 35.52, 35.56])
gate_3 = np.array([48.40, 48.53, 48.68, 48.63])
gate_4 = np.array([61.25, 61.51, 61.56, 61.52])
gate_5 = np.array([74.32, 74.69, 74.48, 74.61])

# Ball Measurements in mm
ball_diameter = [12.685, 12.681, 12.66, 12.73]

# Angle measurements in degrees
angle_incline_1 = [77.5, 77.45, 77.50, 77.80]
angle_incline_2 = [76.50, 77.30, 78.20, 76.20]
big_angle_incline_1 = [75.90, 75.70, 75.70, 75.80]
big_angle_incline_2 = [76.00, 76.05, 76.02, 76.01]

reverse_angle_incline_1 = [76.90, 77.84, 76.80, 76.90]
reverse_angle_incline_2 = [76.10, 77.51, 76.20, 76.50]
reverse_big_angle_incline_1 = [76.80, 76.40, 76.18, 76.15]
reverse_big_angle_incline_2 = [76.90, 76.85, 76.90, 76.85]

pythagoras_1 = 0
pythagoras_2 = 0

### Determine center values and uncertaincies ###
# Gates
central_value_gate_1 = np.sum(gate_1) / 4
central_value_gate_2 = np.sum(gate_2) / 4
central_value_gate_3 = np.sum(gate_3) / 4
central_value_gate_4 = np.sum(gate_4) / 4
central_value_gate_5 = np.sum(gate_5) / 4

# err_gate_1 = np.sqrt(np.)
# err_gate_2 = 
# err_gate_3 = 
# err_gate_4 = 
# err_gate_5 = 

# Angles

### Determining the acceleration ###
gate_positions = np.array([central_value_gate_1, central_value_gate_2, central_value_gate_3, central_value_gate_4, central_value_gate_5])
gate_positions = (gate_positions-gate_positions[0]) / 100
#plt.plot(gate_times, gate_positions)
#plt.show()



err_gate_positions = np.array([0.002, 0.002, 0.002, 0.002, 0.002])




### Chi2 functions for determining the acceleration ###

def fitting_function(x, alpha0, alpha1, alpha2):
    return alpha0 + alpha1*x + alpha2*x**2


def chi2_ball_on_incline(x, y, err_y):
    # Now we define a ChiSquare to be minimised (using probfit), where we set various settings and starting parameters:
    chi2_object = Chi2Regression(fitting_function, x, y, err_y) 
    minuit = Minuit(chi2_object, pedantic=False, alpha0=0.0, alpha1=0.0, alpha2=0.5, print_level=0)  
    minuit.migrad() # Perform the actual fit
    minuit_output = [minuit.get_fmin(), minuit.get_param_states()] # Save the output parameters in case needed
    
    # Here we extract the fitting parameters and their errors
    alpha0_fit = minuit.values['alpha0']
    alpha1_fit = minuit.values['alpha1']
    alpha2_fit = minuit.values['alpha2']
    sigma_alpha0_fit = minuit.errors['alpha0']
    sigma_alpha1_fit = minuit.errors['alpha1']
    sigma_alpha2_fit = minuit.errors['alpha2']

    Npoints = 5 # Number of data points
    Nvar = 3 # Number of variables
    Ndof_fit = Npoints - Nvar    # Number of degrees of freedom = Number of data points - Number of variables
    
    Chi2_fit = minuit.fval                          # The chi2 value
    Prob_fit = stats.chi2.sf(Chi2_fit, Ndof_fit)    # The chi2 probability given N degrees of freedom

    # Plotting
    fig, ax = plt.subplots(figsize=(10,6))
    plotting_times = np.linspace(0.0, 0.6, 1000)
    ax.errorbar(x, y, fmt='ro', ecolor='k', elinewidth=1, capsize=2, capthick=1)
    ax.plot(plotting_times, fitting_function(plotting_times, alpha0_fit, alpha1_fit, alpha2_fit), '-r')
    # Add nice text
    d = {'alpha0':  [alpha0_fit, sigma_alpha0_fit],
        'alpha1':   [alpha1_fit, sigma_alpha1_fit],
        'alpha2':   [alpha2_fit, sigma_alpha2_fit],
        'Chi2':     Chi2_fit,
        'ndf':      Ndof_fit,
        'Prob':     Prob_fit,
        }
    text = nice_string_output(d, extra_spacing=2, decimals=3)
    add_text_to_ax(0.02, 0.95, text, ax, fontsize=14)
    fig.tight_layout()
    plt.show()

    # Print results
    print(f"Acceleration={alpha2_fit} +- {sigma_alpha2_fit}")
    print(f"Chi2={Chi2_fit}, prop={Prob_fit}")

    return alpha2_fit, sigma_alpha2_fit


### Peform Chi2 on the two data sets ###
chi2_ball_on_incline(gate_times, gate_positions, err_gate_positions)
chi2_ball_on_incline(reverse_gate_times, gate_positions, err_gate_positions)

















