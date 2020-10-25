import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
#from pylab import polyfit
from scipy.stats import linregress
import pandas as pd
import sys


hyperparameters_file = sys.argv[1]
initial_mu_file = sys.argv[2]
mu_delta_file = sys.argv[3]
initial_learning_rate_file = sys.argv[4]
learning_rate_delta_file = sys.argv[5]
initial_weight_decay_file = sys.argv[6]
weight_decay_delta_file = sys.argv[7]
alpha_file = sys.argv[8]
velocity_reset_file = sys.argv[9]
input_dropout_file = sys.argv[10]
hidden_dropout_file = sys.argv[11]
batch_size_file = sys.argv[12]

v1 = genfromtxt(hyperparameters_file)
#print v1

#print v1

#print "first column of v1:\n"

def write_file(parameter_name, output_file, v1, first_row):
    min_value = [row[first_row] for row in v1]
    max_value = [row[first_row + 1] for row in v1]
    avg_value = [row[first_row + 2] for row in v1]
    best_value = [row[first_row + 3] for row in v1]

    t = range( len(min_value) )

    '''
    print "t:\n"
    print t
    print "\nMin", parameter_name, ":\n"
    print min_value
    print "\nAvg", parameter_name, ":\n"
    print avg_value
    print "\nBest", parameter_name, ":\n"
    print best_value
    print "\nMax", parameter_name, ":\n"
    print max_value
    '''

    # plot it!
    fig, ax = plt.subplots(1)

    ax.plot(t, avg_value, lw=2, label='Average ' + parameter_name, color='blue')
    ax.plot(t, best_value, lw=2, label='Best ' + parameter_name, color='green')
    ax.fill_between(t, min_value, max_value, facecolor='blue', alpha=0.25)

    #ax.set_yscale('log')
    #ax.set_xscale('log')

    ax.set_title('EXACT Population ' + parameter_name)
    ax.legend(loc='upper left')
    ax.set_xlabel('Genomes Evaluated')
    ax.set_ylabel('Value')
    ax.grid()

    plt.savefig(output_file)

    return

write_file("Initial Mu", initial_mu_file, v1, 0);
write_file("Mu Delta", mu_delta_file, v1, 4);
write_file("Initial Learning Rate", initial_learning_rate_file, v1, 8);
write_file("Learning Rate Delta", learning_rate_delta_file, v1, 12);
write_file("Initial Weight Decay", initial_weight_decay_file, v1, 16);
write_file("Weight Decay Delta", weight_decay_delta_file, v1, 20);

write_file("Alpha", alpha_file, v1, 24);
write_file("Velocity Reset", velocity_reset_file, v1, 28);

write_file("Input Dropout", input_dropout_file, v1, 32);
write_file("Hidden Dropout", hidden_dropout_file, v1, 36);
write_file("Batch Size", batch_size_file, v1, 40);


