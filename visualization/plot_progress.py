import csv
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
#from pylab import polyfit
from scipy.stats import linregress
import pandas as pd
import sys
from cycler import cycler


input_directory = sys.argv[1]
output_directory = sys.argv[2]
search_name = sys.argv[3]

progress_file = input_directory + "/progress.txt"

fitness_filename = output_directory + search_name + "_fitness_progress.png"
epochs_filename = output_directory + search_name + "_epochs_progress.png"
nodes_edges_filename = output_directory + search_name + "_nodes_edges_progress.png"
weights_filename = output_directory + search_name + "_weights_progress.png"

node_percentage_file = output_directory + search_name + "_node_percentage_progress.png"
edge_percentage_file = output_directory + search_name + "_edge_percentage_progress.png"
other_percentage_file = output_directory + search_name + "_other_percentage_progress.png"

f = open(progress_file, 'rb')
reader = csv.reader(f)
headers = reader.next()

#print headers

v1 = genfromtxt(progress_file)
#v2 = genfromtxt('migration_ring_salsa_10_320.txt', delimiter=' ')
#print v1

#print v1

#print "first column of v1:\n"
t = [row[3] for row in v1]
'''
print "t:\n"
print t
'''

def plot_min_avg_max(filename, y_label_title, names, colors, first_row, values, legend_loc = 'upper right', y_min = None, y_max = None):
    fig, ax = plt.subplots(1)

    title = 'EXACT Population';
    position = 0
    for name in names:
        min_value = [row[first_row] for row in values]
        avg_value = [row[first_row + 1] for row in values]
        max_value = [row[first_row + 2] for row in values]

        title += ' ' + name

        ax.plot(t, avg_value, lw=2, label=name, color=colors[position])
        ax.fill_between(t, min_value, max_value, facecolor=colors[position], alpha=0.25)
        first_row += 3
        position += 1

    ax.relim()
    if y_max != None and y_min != None:
        plt.ylim(ymax = y_max, ymin = y_min)

    if y_min != None:
        plt.ylim(ymin = y_min)


    ax.set_title(title)
    ax.legend(loc = legend_loc)
    ax.set_xlabel('Genomes Evaluated')
    ax.set_ylabel(y_label_title)
    ax.grid()

    plt.savefig(filename)
    plt.clf()
    plt.cla()
    plt.close()

plot_min_avg_max(fitness_filename, 'Fitness (CNN best error)', ["Fitness"], ["blue"], 4, v1, legend_loc = 'upper right', y_min = 0, y_max = 2000)
plot_min_avg_max(epochs_filename, 'Epochs to Best Error', ["Epochs"], ["green"], 7, v1, legend_loc = 'lower right', y_min = 0)
plot_min_avg_max(nodes_edges_filename, 'Count', ["Enabled Nodes", "Enabled Pooling Edges", "Enabled Convolutional Edges"], ["blue", "green", "yellow"], 10, v1, legend_loc = 'upper left', y_min = 0)
plot_min_avg_max(weights_filename, 'Count', ["Weights"], ["green"], 19, v1, legend_loc = 'upper left', y_min = 0)

n_initial_columns = 23

def generate_percentage_plot(output_filename, percentage_type, title):
    num_plots = len(headers) - n_initial_columns
    colormap = plt.cm.gist_ncar

    plt.rc('axes', prop_cycle=(cycler('color', [colormap(i) for i in np.linspace(0, 0.9, num_plots)]) + cycler('linestyle', ['-', '-', '-', '-', '--', '--', '--', '--', ':', ':', ':',  '-.', '-.', '-.'])))
    fig, ax = plt.subplots(1)


    for row_number in range(n_initial_columns, len(headers)):
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

    ax.set_title(title)
#ax.legend(loc='upper right')
    ax.set_xlabel('Genomes Evaluated')
    ax.set_ylabel('Genomes Inserted (%)')
    ax.grid()

    plt.savefig(output_filename)

generate_percentage_plot(node_percentage_file, "node", "Node Mutation Operation Statistics")
generate_percentage_plot(edge_percentage_file, "edge", "Edge Mutation Operation Statistics")
generate_percentage_plot(other_percentage_file, "", "Other Mutation Operation Statistics")
