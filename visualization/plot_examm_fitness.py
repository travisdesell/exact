import csv
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
#from pylab import polyfit
from scipy.stats import linregress
import pandas as pd
import sys
from cycler import cycler
import glob
import statistics
import math
from natsort import natsorted, ns

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 24


plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

if len(sys.argv) < 7:
    print "Incorrect usage!"
    print "Run with:"
    print "python " + sys.argv[0] + "<min y value> <max y value> <x type> <chart title> <output filename> <directories to use>"
    print "note that each directory listed should have 0, 1, 2, .. n as subdirectories each with a fitness_log.csv, which this script will use to generate the min/avg/max fitnesses"
    print "x_type can be: 'inserted', 'epochs' or 'time'"
    exit(1)



y_min = float(sys.argv[1])
y_max = float(sys.argv[2])
x_type = sys.argv[3]
chart_title = sys.argv[4]
figure_filename = sys.argv[5]

if not (x_type == "time" or x_type == "inserted" or x_type == "epochs"):
    print "Incorrect x_type: {}".format(x_type)
    print "x_type can be: 'inserted', 'epochs' or 'time'"
    exit(1)


base_directories = sys.argv[6:]
print "base_directories:"
print base_directories

base_directories = natsorted(base_directories, alg=ns.IGNORECASE)

print "sorted base directories:"
print base_directories

colors = ["red", "darkblue", "lightblue", "green", "purple", "plum", "orange", "grey", "darkgrey", "pink", "magenta", "cyan", "lightgreen", "tomato", "chocolate", "saddlebrown", "peachpuff", "yellowgreen", "orchid", "mediumpurple", "seagreen", "palevioletred"]
current_fold = 0
fig, ax = plt.subplots(1, figsize=(20,10))

for run_directory in base_directories:
    print "run directory: '" + run_directory + "'"
    run_name = run_directory.split("/")[-1]
    print "run name is: '" + run_name + "'"

    if ".csv" in run_directory:
        continue

    #get the values for each of these from each
    #repeat folder (0..n) in the run directory
    inserted_genomes_list = []
    backprop_epochs_list = []
    time_seconds_list = []
    best_mse_list = []

    for input_file in sorted(glob.glob(run_directory + "/*/fitness_log.csv")):
        print "log file: '" + input_file + "'"

        inserted_genomes = []
        backprop_epochs = []
        time_seconds = []
        best_mse = []
        with open(input_file) as fp:
           line = fp.readline()
           print "headers: {}".format(line)

           line = fp.readline()
           while line:
               values = line.split(",")
               inserted_genomes.append(int(values[0]))
               backprop_epochs.append(int(values[1]))
               time_seconds.append(int(values[2]))
               best_mse.append(float(values[4]))

               line = fp.readline()

        inserted_genomes_list.append(inserted_genomes)
        backprop_epochs_list.append(backprop_epochs)
        time_seconds_list.append(time_seconds)
        best_mse_list.append(best_mse)

    #get these values over each list
    merged_inserted_genomes = []
    merged_time_seconds = []
    merged_backprop_epochs = []

    merged_min = []
    merged_avg = []
    merged_max = []

    number_log_files = len(best_mse_list)

    #find the shortest and longest runs, as the fitness logs may not be all the same length
    print "number of log files for this directory:", number_log_files
    x_min = len(best_mse_list[0])
    x_max = len(best_mse_list[0])
    for j in range(0, number_log_files):
        if len(best_mse_list[j]) < x_min:
            x_min = len(best_mse_list[j])
        if len(best_mse_list[j]) > x_max:
            x_max = len(best_mse_list[j])
        print "\trow:", len(best_mse_list[j])
    
    print "shortest length: ", x_min
    print "longest length: ", x_max

    time_skips = []

    for j in range(0, x_max):
        min_mse = 500000;
        max_mse = 0;
        avg_mse = 0;
        avg_time = 0;

        avg_inserted_genomes = 0
        avg_backprop_epochs = 0

        #use the count of actually used values to calculate averages
        count = 0
        for i in range(0, number_log_files):
            #skip this sequence if it was not long enough
            if len(best_mse_list[i]) <= j:
                #print "j: ", j, ", len(best_mse_list[i]): ", len(best_mse_list[i]), " -- SKIPPING!"
                continue
            #else:
                #print "j: ", j, ", len(best_mse_list[i]): ", len(best_mse_list[i])

            if j > 0:
                #track time skips because cluster was stopping and restarting
                #runs which messes with wallclock time
                time_skip = time_seconds_list[i][j] - time_seconds_list[i][j-1]
                if math.isnan(time_skip):
                    print "nan time skip, time_seconds_list[{}][{}]: {}, time_seconds_list[{}][{}]: {}".format(i, j, time_seconds_list[i][j], i, j-1, time_seconds_list[i][j-1])
                else:
                    time_skips.append(time_skip)
                
            count = count + 1

            value = best_mse_list[i][j]

            avg_time += time_seconds_list[i][j]
            avg_inserted_genomes += inserted_genomes_list[i][j]
            avg_backprop_epochs += backprop_epochs_list[i][j]

            if value < min_mse:
                min_mse = value
            if value > max_mse:
                max_mse = value

            avg_mse += value

        merged_min.append(min_mse)
        merged_max.append(max_mse)

        merged_avg.append(avg_mse /count)

        merged_inserted_genomes.append(avg_inserted_genomes / count)
        merged_backprop_epochs.append(avg_backprop_epochs / count)
        merged_time_seconds.append((avg_time / count) / 1000.0) #convert from ms to s


    min_time_skip = min(time_skips)
    max_time_skip = max(time_skips)
    avg_time_skip = sum(time_skips)/len(time_skips)
    median_time_skip = statistics.median(time_skips)

    print "time skip count: ", len(time_skips)
    print "time skip min: ", min_time_skip, ", time skip avg: ", avg_time_skip, ", max_time_skip: ", max_time_skip, ", median_time_skip: ", median_time_skip

    clip_runs = 1
    if clip_runs:
        merged_min = merged_min[0:x_min]
        merged_avg = merged_avg[0:x_min]
        merged_max = merged_max[0:x_min]
        merged_inserted_genomes = merged_inserted_genomes[0:x_min]
        merged_backprop_epochs = merged_backprop_epochs[0:x_min]
        merged_time_seconds = merged_time_seconds[0:x_min]

    print len(merged_min)
    print len(merged_avg)
    print len(merged_max)
    print len(merged_inserted_genomes)
    print len(merged_backprop_epochs)
    print len(merged_time_seconds)

    print colors
    print current_fold
    print colors[current_fold]

    linestyle = "solid" #default
    if "cpu_108" in run_name:
        linestyle = "dotted"
    elif "cpu_216" in run_name:
        linestyle = "dashdot"
    elif "cpu_432" in run_name:
        linestyle = "dashed"

    if x_type == "time":
        ax.plot(merged_time_seconds, merged_avg, lw=1, label=run_name, color=colors[current_fold], linestyle=linestyle)
        ax.fill_between(merged_time_seconds, merged_max, merged_min, facecolor=colors[current_fold], alpha=0.25)
    elif x_type == "inserted":
        ax.plot(merged_inserted_genomes, merged_avg, lw=1, label=run_name, color=colors[current_fold], linestyle=linestyle)
        ax.fill_between(merged_inserted_genomes, merged_max, merged_min, facecolor=colors[current_fold], alpha=0.25)
    elif x_type == "epochs":
        ax.plot(merged_backprop_epochs, merged_avg, lw=1, label=run_name, color=colors[current_fold], linestyle=linestyle)
        ax.fill_between(merged_backprop_epochs, merged_max, merged_min, facecolor=colors[current_fold], alpha=0.25)

    current_fold = current_fold + 1


ax.relim()
#ax.set_yscale('log')
plt.ylim(ymax = y_max, ymin = y_min)


ax.set_title(chart_title)
ax.legend(loc = 'upper right')

if x_type == "time":
    ax.set_xlabel('Wallclock Time (seconds)')
elif x_type == "inserted":
    ax.set_xlabel('Evaluated RNNs')
elif x_type == "epochs":
    ax.set_xlabel('Total Backpropagation Epochs')


ax.set_ylabel('Best MSE')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.82, box.height])
leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#ax.legend(loc='center right')

ax.grid()

print "saving figure as '" + figure_filename + "'"


plt.savefig(figure_filename, bbox_extra_artists=(leg,), bbox_inches='tight')
plt.clf()
plt.cla()
plt.close()
