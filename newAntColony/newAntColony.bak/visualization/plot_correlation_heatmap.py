import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
#from pylab import polyfit
from scipy.stats import linregress
import pandas as pd
import sys


fitness_hyperparameters_file = sys.argv[1]

v1 = genfromtxt(fitness_hyperparameters_file,delimiter=',', names=True)
#print v1

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

fitness = [row[0] for row in v1]

initial_mu = [row[1] for row in v1]
mu_delta = [row[2] for row in v1]
initial_learning_rate = [row[3] for row in v1]
learning_rate_delta = [row[4] for row in v1]
initial_weight_decay = [row[5] for row in v1]
weight_decay_delta = [row[6] for row in v1]

alpha = [row[7] for row in v1]
batch_size = [row[8] for row in v1]
velocity_reset = [row[9] for row in v1]

input_dropout_probabilty = [row[10] for row in v1]
hidden_dropout_probabilty = [row[11] for row in v1]

names = ["Fitness", "Initial Mu", "Mu Delta", "Initial Learning Rate", "Learning Rate Delta", "Initial Weight Decay", "Weight Decay Delta", "Alpha", "Batch Size", "Velocity Reset", "Input Dropout Prob.", "Hidden Dropout Prob."]

values = []

for i in range(0, len(v1[0])):
    rv = np.array([row[i] for row in v1])
    print "average of row '{0}' is: {1}".format(names[i], np.mean(rv))
    print "std of row '{0}' is: {1}".format(names[i], np.std(rv))

    rv = (rv - np.mean(rv)) / np.std(rv)
    values.append(rv)

values = np.array(values)

print names
print values

correlations = []
for i in range(0, len(v1[0])):
    r = []
    for j in range(0, len(v1[0])):
        r.append(np.correlate(values[i], values[j])[0] / len(v1))
    correlations.append(r)

print correlations

fig, ax = plt.subplots()
heatmap = plt.pcolor(correlations, cmap='RdBu', alpha = 0.8, vmin=-1, vmax=1)
plt.colorbar()

ax.set_xticks(np.arange(len(v1[0])) + 0.5, minor = False)
ax.set_yticks(np.arange(len(v1[0])) + 0.5, minor = False)

ax.invert_yaxis()
ax.xaxis.tick_top()

ax.set_xticklabels(names, minor = False)
ax.set_yticklabels(names, minor = False)

ax.set_frame_on(False)
ax.grid(False)
ax = plt.gca()
plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig("hyperparameter_correlations.png")

