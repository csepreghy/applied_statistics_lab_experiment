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

    prop_err_t_gate_1 = np.sqrt(np.sum((err_t_gate_1-np.mean(err_t_gate_1))**2))
    prop_err_t_gate_2 = np.sqrt(np.sum((err_t_gate_2-np.mean(err_t_gate_2))**2))
    prop_err_t_gate_3 = np.sqrt(np.sum((err_t_gate_3-np.mean(err_t_gate_3))**2))
    prop_err_t_gate_4 = np.sqrt(np.sum((err_t_gate_4-np.mean(err_t_gate_4))**2))
    prop_err_t_gate_5 = np.sqrt(np.sum((err_t_gate_5-np.mean(err_t_gate_5))**2))
    err_gate_times = np.array([prop_err_t_gate_1, prop_err_t_gate_2, prop_err_t_gate_3, prop_err_t_gate_4, prop_err_t_gate_5])
    print()
    print("Weighted Average times")
    print(f"{gate_times} +- {err_gate_times}")
    print()
    return gate_times, err_gate_times


### Data ###
# Importing date from the experiments as weigthed averages.
gate_times, err_gate_times = collect_gate_times('../data/Measurements_1')
reverse_gate_times, reverse_err_gate_times = collect_gate_times('../data/Measurements_2')

# Path measurements in m
gate_1 = np.array([22.28, 22.45, 22.52, 22.42]) /100
gate_2 = np.array([35.36, 35.62, 35.52, 35.56]) /100
gate_3 = np.array([48.40, 48.53, 48.68, 48.63]) /100
gate_4 = np.array([61.25, 61.51, 61.56, 61.52]) /100
gate_5 = np.array([74.32, 74.69, 74.48, 74.61]) /100

# Ball measurements in m
ball_diameter = np.array([12.685, 12.681, 12.66, 12.73]) /1000

# Slide measurements
slide_width = np.array([0])

# Angle measurements in degrees
angle_incline_1 = np.array([77.5, 77.45, 77.50, 77.80])
angle_incline_2 = np.array([76.50, 77.30, 78.20, 76.20])
reverse_big_angle_incline_1 = np.array([75.90, 75.70, 75.70, 75.80])
reverse_big_angle_incline_2 = np.array([76.00, 76.05, 76.02, 76.01])

reverse_angle_incline_1 = np.array([76.90, 77.84, 76.80, 76.90])
reverse_angle_incline_2 = np.array([76.10, 77.51, 76.20, 76.50])
big_angle_incline_1 = np.array([76.80, 76.40, 76.18, 76.15])
big_angle_incline_2 = np.array([76.90, 76.85, 76.90, 76.85])

# Pythagoras angle measurements in cm
pythagoras_kat_1 = np.array([90.20, 90.35, 90.45, 90.05])
pythagoras_kat_2 = np.array([22.15, 22.25, 22.16, 22.19])
pythagoras_hypo = np.sqrt(pythagoras_kat_1**2 + pythagoras_kat_2**2)


### Determine center values and uncertaincies ###
### Gates
print("\n### Gate measurements ###")
central_value_gate_1 = np.sum(gate_1) / 4
central_value_gate_2 = np.sum(gate_2) / 4 - central_value_gate_1
central_value_gate_3 = np.sum(gate_3) / 4 - central_value_gate_1
central_value_gate_4 = np.sum(gate_4) / 4 - central_value_gate_1
central_value_gate_5 = np.sum(gate_5) / 4 - central_value_gate_1
central_value_gate_1 = np.sum(gate_1) / 4 - central_value_gate_1
err_gate_1 = np.sqrt((np.sum((gate_1-np.mean(gate_1))**2)/4)) /2
err_gate_2 = np.sqrt((np.sum((gate_2-np.mean(gate_2))**2)/4)) /2
err_gate_3 = np.sqrt((np.sum((gate_3-np.mean(gate_3))**2)/4)) /2
err_gate_4 = np.sqrt((np.sum((gate_4-np.mean(gate_4))**2)/4)) /2
err_gate_5 = np.sqrt((np.sum((gate_5-np.mean(gate_5))**2)/4)) /2
# Add to array
gate_positions = np.array([central_value_gate_1, central_value_gate_2, central_value_gate_3, central_value_gate_4, central_value_gate_5])
err_gate_positions = np.array([err_gate_1, err_gate_2, err_gate_3, err_gate_4, err_gate_5])
print("Gate positions")
print(f"({gate_positions} +- {err_gate_positions}) m")

### Ball diameter
print("\n### Ball Measurements ###")
central_value_ball_diameter = np.mean(ball_diameter)
err_ball_diameter = np.sqrt((np.sum((ball_diameter-np.mean(ball_diameter))**2)/4)) /2
print("Ball diameter:")
print(f"({central_value_ball_diameter} +- {err_ball_diameter}) m")

### Slide width
print("\n### Slide width Measurements ###")
central_value_slide_width = np.mean(slide_width)
err_slide_width = np.sqrt((np.sum((slide_width-np.mean(slide_width))**2)/4)) /2
print("Slide width:")
print(f"({central_value_slide_width} +- {err_slide_width}) m")

### Angles
# Normal setup
# The measured angle is in the "wrong" corner
print("\n### Angle measurements ###")
theta = 90.0 - (angle_incline_1+angle_incline_2)/2
theta_big = 90.0 - (big_angle_incline_1+big_angle_incline_2)/2
# determine the errors
err_theta = np.sqrt((np.sum((theta-np.mean(theta))**2)/4)) /2
err_theta_big = np.sqrt((np.sum((theta_big-np.mean(theta_big))**2)/4)) /2
print("Normal setup:")
print(f"Small: {theta} +- {err_theta}")
print(f"Big:   {theta_big} +- {err_theta_big}")
# Take weighted average
weighted_average_theta = (np.sum(theta/err_theta**2) + np.sum(theta_big/err_theta_big**2))/( 4*np.sum(1/err_theta**2) + 4*np.sum(1/err_theta_big**2))
err_weighted_average_theta = np.sqrt(1 / ( 4*np.sum(1/err_theta**2) + 4*np.sum(1/err_theta_big**2)) )
print("Weighted average theta:")
print(f"{weighted_average_theta} +- {err_weighted_average_theta}")

# Reverse setup
reverse_theta = 90.0 - (reverse_angle_incline_1+reverse_angle_incline_2)/2
reverse_theta_big = 90.0 - (reverse_big_angle_incline_1+reverse_big_angle_incline_2)/2
# Determine the errors
reverse_err_theta = np.sqrt((np.sum((reverse_theta-np.mean(reverse_theta))**2)/4)) /2
reverse_err_theta_big = np.sqrt((np.sum((reverse_theta_big-np.mean(reverse_theta_big))**2)/4)) /2
print("\nReverse setup:")
print(f"Small: {theta} +- {err_theta}")
print(f"Big:   {theta_big} +- {err_theta_big}")
# Take weighted average
reverse_weighted_average_theta = (np.sum(reverse_theta/reverse_err_theta**2) + np.sum(reverse_theta_big/reverse_err_theta_big**2))/( 4*np.sum(1/reverse_err_theta**2) + 4*np.sum(1/reverse_err_theta_big**2))
reverse_err_weighted_average_theta = np.sqrt(1 / ( 4*np.sum(1/reverse_err_theta**2) + 4*np.sum(1/reverse_err_theta_big**2)) )
print("Reverse weighted average theta:")
print(f"{reverse_weighted_average_theta} +- {reverse_err_weighted_average_theta}")

# Determine the Table angle
print("\nTable angle:")
table_angle = abs((weighted_average_theta - reverse_weighted_average_theta) /2)
err_table_angle = np.sqrt(0.25*err_weighted_average_theta**2 + 0.25*reverse_err_weighted_average_theta**2)
print(f"{table_angle} +- {err_table_angle}")
print("\"True\" theta")
true_theta = np.array([weighted_average_theta + table_angle, reverse_weighted_average_theta - table_angle])
err_true_theta = np.array([np.sqrt(err_weighted_average_theta**2 + err_table_angle**2), np.sqrt(reverse_err_weighted_average_theta**2 + err_table_angle**2)])
print(f"{true_theta} +- {err_true_theta}, [original, reverse]")

print("\nPythagoras \"True\" theta")
theta_pyth = np.degrees(np.arctan(pythagoras_kat_2/pythagoras_kat_1))
central_value_theta_pyth = np.sum(theta_pyth) /4
err_theta_pyth = np.sqrt((np.sum((theta_pyth-np.mean(theta_pyth))**2)/4)) /2
print(f"{central_value_theta_pyth} +- {err_theta_pyth}")


### Determining the acceleration ###

# Chi2 functions for determining the acceleration
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

    Npoints = np.shape(x)[0] # Number of data points
    Nvar = 3 # Number of variables
    Ndof_fit = Npoints - Nvar    # Number of degrees of freedom = Number of data points - Number of variables
    
    Chi2_fit = minuit.fval                          # The chi2 value
    Prob_fit = stats.chi2.sf(Chi2_fit, Ndof_fit)    # The chi2 probability given N degrees of freedom

    # Plotting
    #fig, ax = plt.subplots(figsize=(10,6))
    #plotting_times = np.linspace(0.0, 0.6, 1000)
    #ax.errorbar(x, y, fmt='ro', ecolor='k', elinewidth=1, capsize=2, capthick=1)
    #ax.plot(plotting_times, fitting_function(plotting_times, alpha0_fit, alpha1_fit, alpha2_fit), '-r')
    #ax.set_xlim(0.520,0.527)
    #ax.set_ylim(0.5,0.6)
    # Add nice text
    #d = {'alpha0':  [alpha0_fit, sigma_alpha0_fit],
    #    'alpha1':   [alpha1_fit, sigma_alpha1_fit],
    #    'alpha2':   [alpha2_fit, sigma_alpha2_fit],
    #    'Chi2':     Chi2_fit,
    #    'ndf':      Ndof_fit,
    #    'Prob':     Prob_fit,
    #    }
    #text = nice_string_output(d, extra_spacing=2, decimals=3)
    #add_text_to_ax(0.02, 0.95, text, ax, fontsize=14)
    #fig.tight_layout()
    #plt.show()

    # Print results
    #print(f"Acceleration={alpha2_fit} +- {sigma_alpha2_fit}")
    #print(f"Chi2={Chi2_fit}, prop={Prob_fit}")

    return alpha2_fit, sigma_alpha2_fit

print("\n### Determine aceleration ###")
# Acceleration for each experiment
acc = np.zeros(25)
err_acc = np.zeros(25)
for i, filename in enumerate(os.listdir('../data/Measurements_1')):
    path = os.path.join('../data/Measurements_1', filename)
    time, voltage = read_ball_data(path)
    t_gates, err_t_gates = time_at_gate(time, voltage)
    acc[i], err_acc[i] = chi2_ball_on_incline(t_gates, gate_positions, err_gate_positions)

rev_acc = np.zeros(25)
rev_err_acc = np.zeros(25)
for i, filename in enumerate(os.listdir('../data/Measurements_2')):
    path = os.path.join('../data/Measurements_2', filename)
    time, voltage = read_ball_data(path)
    t_gates, err_t_gates = time_at_gate(time, voltage)
    rev_acc[i], rev_err_acc[i] = chi2_ball_on_incline(t_gates, gate_positions, err_gate_positions)

# Weighted average
weighted_average_acc = np.sum(acc/err_acc**2) / np.sum(1/err_acc**2)
weighted_err_acc = 1 / np.sum(1/err_acc**2)
rev_weighted_average_acc = np.sum(rev_acc/rev_err_acc**2) / np.sum(1/rev_err_acc**2)
rev_weighted_err_acc = 1 / np.sum(1/rev_err_acc**2)
print("\nWeighted average og the accelerations")
print(weighted_average_acc)
print(rev_weighted_average_acc)

# Determine acc from the weigthed times
wt_a, wt_err_a = chi2_ball_on_incline(gate_times, gate_positions, err_gate_positions)
rev_wt_a, rev_wt_err_a = chi2_ball_on_incline(reverse_gate_times, gate_positions, err_gate_positions)
print("\nacceleration determined from the weighted times")
print(wt_a)
print(rev_wt_a)


### Calculate G ###
print("\n### Determine g ###")

def ball_on_incline_g(a, theta, delta_theta, D_ball, d_rail, err_a, err_theta, err_delta_theta, err_D_ball, err_d_rail, exp_type):
    # Function calculating g and propergating errors
    # Give delta_theta correct sign
    if exp_type == 'reverse':
        delta_theta = delta_theta
    elif exp_type == 'original':
        delta_theta = -delta_theta
    else:
        return print("You need to specify if the slide was in \'original\' position or in \'reverse\' position")
    # Convert angles into radians
    theta = np.radians(theta)
    delta_theta = np.radians(delta_theta)
    err_theta = np.radians(err_theta)
    err_delta_theta = np.radians(err_delta_theta)
    # Define "constants"
    acc_angle = a / np.sin(theta+delta_theta)
    Dd = 1 + (2*D_ball**2)/(5*(D_ball**2 - d_rail**2))
    
    # Calculate g
    g = acc_angle * Dd

    # Define derivatives
    dg_da = Dd/np.sin(theta+delta_theta)
    dg_dtheta = Dd*a*np.cos(theta+delta_theta)/np.sin(theta+delta_theta)**2
    dg_ddtheta = dg_dtheta
    dg_dD = acc_angle*4*D_ball*d_rail**2 / (5*(D_ball**2 - d_rail**2)**2)
    dg_dd = acc_angle*4*D_ball**2*d_rail / (5*(D_ball**2 - d_rail**2)**2)

    # Add squared derivatives and errors to arrays
    derivatives_squared = np.array([dg_da**2, dg_dtheta**2, dg_ddtheta**2, dg_dD**2, dg_dd**2])
    errors_squared = np.array([err_a**2, err_theta**2, err_delta_theta**2, err_D_ball**2, err_d_rail**2])
    # Calculate error on g
    err_g = np.sqrt(np.sum(derivatives_squared*errors_squared))

    return g, err_g

# Troel number for slide width
central_value_slide_width = 0.006
err_slide_width = 0.0001

#Calculating g from weighted average of the times
wt_g, wt_err_g = ball_on_incline_g(wt_a, true_theta[0], table_angle, central_value_ball_diameter, central_value_slide_width, wt_err_a, err_true_theta[0], err_table_angle, err_ball_diameter, err_slide_width, exp_type='original')
rev_wt_g, rev_wt_err_g = ball_on_incline_g(rev_wt_a, true_theta[1], table_angle, central_value_ball_diameter, central_value_slide_width, rev_wt_err_a, err_true_theta[1], err_table_angle, err_ball_diameter, err_slide_width, exp_type='reverse')
print("\nCalulated g from weighted average of times")
print(np.sin(np.radians(true_theta[0]-table_angle)))
print(np.sin(np.radians(true_theta[0]+table_angle)))
print(f"({wt_g} +- {wt_err_g}) m/s^2")
print(f"({rev_wt_g} +- {rev_wt_err_g}) m/s^2")

#Calculating g from weighted average of the accelerations
wa_g, wa_err_g = ball_on_incline_g(weighted_average_acc, true_theta[0], table_angle, central_value_ball_diameter, central_value_slide_width, weighted_err_acc, err_true_theta[0], err_table_angle, err_ball_diameter, err_slide_width, exp_type='original')
rev_wa_g, rev_wa_err_g = ball_on_incline_g(rev_weighted_average_acc, true_theta[1], table_angle, central_value_ball_diameter, central_value_slide_width, rev_weighted_err_acc, err_true_theta[0], err_table_angle, err_ball_diameter, err_slide_width, exp_type='reverse')
print("\nCalulated g from weighted average of accelerations")
print(np.sin(np.radians(true_theta[0]-table_angle)))
print(np.sin(np.radians(true_theta[0]+table_angle)))
print(f"({wa_g} +- {wa_err_g}) m/s^2")
print(f"({rev_wa_g} +- {rev_wa_err_g}) m/s^2")

#Calculating g for each acceleration
a_g = np.zeros(25)
rev_a_g = np.zeros(25)
err_a_g = np.zeros(25)
rev_err_a_g = np.zeros(25)
for i in range(25):
    a_g[i], err_a_g[i] = ball_on_incline_g(acc[i], true_theta[0], table_angle, central_value_ball_diameter, central_value_slide_width, err_acc[i], err_true_theta[0], err_table_angle, err_ball_diameter, err_slide_width, exp_type='original')
    rev_a_g[i], rev_err_a_g[i] = ball_on_incline_g(rev_acc[i], true_theta[1], table_angle, central_value_ball_diameter, central_value_slide_width, rev_err_acc[i], err_true_theta[1], err_table_angle, err_ball_diameter, err_slide_width, exp_type='reverse')
print("\nCalulated weighted average of g for each acceleration")
print(np.sin(np.radians(true_theta[0]-table_angle)))
print(np.sin(np.radians(true_theta[0]+table_angle)))
print(f"({wa_g} +- {wa_err_g}) m/s^2")
print(f"({rev_wa_g} +- {rev_wa_err_g}) m/s^2")
