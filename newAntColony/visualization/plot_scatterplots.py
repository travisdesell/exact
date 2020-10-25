import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
#from pylab import polyfit
from scipy.stats import linregress
import pandas as pd
import sys


fitness_hyperparameters_file = sys.argv[1]

v1 = genfromtxt(fitness_hyperparameters_file,delimiter=',', names=True)
print v1

#print v1

#print "first column of v1:\n"

def write_file(parameter_name, output_file, v1, first_row):
    fitness = [row[0] for row in v1]
    value = [row[first_row] for row in v1]

    print parameter_name
    print value
    print "\n\n\n"

    # plot it!
    plt.plot(fitness, value, "o", alpha=0.5)

    #fig, ax = plt.subplots(1)

    plt.title(parameter_name + ' vs. Gen+Test Error')
    #ax.legend(loc='upper left')
    plt.xlabel('Fitness')
    plt.ylabel(parameter_name)
    plt.grid()

    plt.savefig(output_file)
    plt.clf()

    return

write_file("Initial Mu", "initial_mu_scatterplot.png", v1, 1);
write_file("Mu Delta", "mu_delta_scatterplot.png", v1, 2);
write_file("Initial Learning Rate", "initial_learning_rate_scatterplot.png", v1, 3);
write_file("Learning Rate Delta", "learning_rate_delta_scatterplot.png", v1, 4);
write_file("Initial Weight Decay", "initial_weight_decay_scatterplot.png", v1, 5);
write_file("Weight Decay Delta", "weight_decay_delta_scatterplot.png", v1, 6);

write_file("Alpha", "alpha_scatterplot.png", v1, 7);
write_file("Batch Size", "batch_size_scatterplot.png", v1, 8);
write_file("Velocity Reset", "velocity_reset_scatterplot.png", v1, 9);

write_file("Input Dropout", "input_dropout_scatterplot.png", v1, 10);
write_file("Hidden Dropout", "hidden_dropout_scatterplot.png", v1, 11);
