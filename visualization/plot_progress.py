import csv
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
#from pylab import polyfit
from scipy.stats import linregress
import pandas as pd
import sys
from cycler import cycler


progress_file = sys.argv[1]
fitness_file = sys.argv[2]
epochs_file = sys.argv[3]
node_percentage_file = sys.argv[4]
edge_percentage_file = sys.argv[5]
other_percentage_file = sys.argv[6]

f = open(progress_file, 'rb')
reader = csv.reader(f)
headers = reader.next()

#print headers

v1 = genfromtxt(progress_file)
#v2 = genfromtxt('migration_ring_salsa_10_320.txt', delimiter=' ')
#print v1

#print v1

#print "first column of v1:\n"

t = [row[2] for row in v1]
min_fitness = [row[3] for row in v1]
avg_fitness = [row[4] for row in v1]
max_fitness = [row[5] for row in v1]

'''
print "t:\n"
print t
print "\n\nmin_fitness:\n"
print min_fitness
print "\n\navg_fitness:\n"
print avg_fitness
print "\n\nmax_fitness:\n"
print max_fitness
'''

# plot it!
fig, ax = plt.subplots(1)
plt.ylim(ymax = 5000, ymin = 0)


ax.plot(t, avg_fitness, lw=2, label='Population Fitness', color='blue')
ax.fill_between(t, min_fitness, max_fitness, facecolor='blue', alpha=0.25)

#ax.set_yscale('log')
#ax.set_xscale('log')

ax.set_title('EXACT Population Fitness')
ax.legend(loc='upper right')
ax.set_xlabel('Genomes Evaluated')
ax.set_ylabel('Fitness (CNN best error)')
ax.grid()

plt.savefig(fitness_file)
plt.clf()
plt.cla()
plt.close()

min_epochs = [row[6] for row in v1]
avg_epochs = [row[7] for row in v1]
max_epochs = [row[8] for row in v1]

'''
print "t:\n"
print t
print "\n\nmin_epochs:\n"
print min_epochs
print "\n\navg_epochs:\n"
print avg_epochs
print "\n\nmax_epochs:\n"
print max_epochs
'''

# plot it!
fig, ax = plt.subplots(1)


ax.plot(t, avg_epochs, lw=2, label='Population Epochs', color='green')
ax.fill_between(t, min_epochs, max_epochs, facecolor='green', alpha=0.25)

#ax.set_yscale('log')
#ax.set_xscale('log')

ax.set_title('EXACT Population Epochs')
ax.legend(loc='lower right')
ax.set_xlabel('Genomes Evaluated')
ax.set_ylabel('Epochs Required')
ax.grid()

plt.savefig(epochs_file)
plt.clf()
plt.cla()
plt.close()


def generate_percentage_plot(output_filename, percentage_type):
    num_plots = len(headers) - 9
    colormap = plt.cm.gist_ncar

    plt.rc('axes', prop_cycle=(cycler('color', [colormap(i) for i in np.linspace(0, 0.9, num_plots)]) + cycler('linestyle', ['-', '-', '-', '-', '--', '--', '--', '--', ':', ':', ':', ':', '-.', '-.', '-.'])))
    fig, ax = plt.subplots(1)


    for row_number in range(9, len(headers)):
        if percentage_type == "":
            if "node" in headers[row_number] or "edge" in headers[row_number]:
                continue
        else:
            if percentage_type not in headers[row_number]:
                continue

        current_row = [ row[row_number] for row in v1]

        ax.plot(t, current_row, lw=1, label=headers[row_number])

#ax.set_yscale('log')
#ax.set_xscale('log')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
        box.width, box.height * 0.9])

# Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=5, prop={'size':8})

    ax.set_title('EXACT Inserted Genome Statistics')
#ax.legend(loc='upper right')
    ax.set_xlabel('Genomes Evaluated')
    ax.set_ylabel('Genomes Inserted (%)')
    ax.grid()

    plt.savefig(output_filename)

generate_percentage_plot(node_percentage_file, "node")
generate_percentage_plot(edge_percentage_file, "edge")
generate_percentage_plot(other_percentage_file, "")
