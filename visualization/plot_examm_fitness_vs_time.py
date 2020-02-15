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



SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


base_directory = sys.argv[1]
print "base_directory: '" + base_directory + "'"
y_max = float(sys.argv[2])
y_min = float(sys.argv[3])
x_max = int(sys.argv[4])
chart_title = sys.argv[5]
figure_filename = sys.argv[6]

colors = ["red", "darkblue", "lightblue", "green", "purple", "plum", "orange", "grey", "darkgrey", "pink", "magenta", "cyan", "lightgreen"]
current_fold = 0
fig, ax = plt.subplots(1, figsize=(20,10))

for run_directory in sorted(glob.glob(base_directory)):
    print "run directory: '" + run_directory + "'"
    run_name = run_directory.split("/")[-1]
    print "run name is: '" + run_name + "'"

    if ".csv" in run_directory:
        continue

    ts = []
    time_seconds_list = []
    best_mses = []

    for input_file in sorted(glob.glob(run_directory + "/*/fitness_log.csv")):
        print "log file: '" + input_file + "'"

        csv_file = open(input_file, 'rb')
        reader = csv.reader(csv_file, delimiter=',')

        headers = reader.next()
        print headers

        v1 = genfromtxt(input_file, delimiter=',')

        t = [row[1] for row in v1]
        time_seconds = [row[2] for row in v1]
        #best_mae = [row[3] for row in v1]
        best_mse = [row[4] for row in v1]
        #print t
        #print best_mse

        time_seconds_list.append(time_seconds)
        ts.append(t)
        best_mses.append(best_mse)

    merged_min = []
    merged_avg = []
    merged_max = []

    merged_time_seconds = []

    print "len(best_mses):", len(best_mses)
    for j in range(0, len(best_mses)):
        best_mses[j] = best_mses[j][0:x_max]
        print "\trow:", len(best_mses[j])

    for j in range(0, len(best_mses[0])):
        min_mse = 500000;
        max_mse = 0;
        avg_mse = 0;
        avg_time = 0;

        for i in range(0, len(best_mses)):
            value = best_mses[i][j]
            avg_time += time_seconds_list[i][j]
            if value < min_mse:
                min_mse = value
            if value > max_mse:
                max_mse = value
            avg_mse += value

        avg_mse /= len(best_mses)

        merged_time_seconds.append((avg_time / len(best_mses)) / 1000.0) #convert from ms to s

        merged_min.append(min_mse)
        merged_max.append(max_mse)
        merged_avg.append(avg_mse)

    merged_min = merged_min[0:x_max]
    merged_avg = merged_avg[0:x_max]
    merged_max = merged_max[0:x_max]
    t = t[0:x_max]
    merged_time_seconds = merged_time_seconds[0:x_max]

    print len(merged_min)
    print len(merged_avg)
    print len(merged_max)
    print len(t)
    print colors
    print current_fold
    print colors[current_fold]

    #ax.plot(t, merged_avg, lw=1, label=run_name, color=colors[current_fold])
    ax.plot(merged_time_seconds, merged_avg, lw=1, label=run_name, color=colors[current_fold])
    #ax.fill_between(t, merged_max, merged_min, facecolor=colors[current_fold], alpha=0.25)
    ax.fill_between(merged_time_seconds, merged_max, merged_min, facecolor=colors[current_fold], alpha=0.25)
    current_fold = current_fold + 1


ax.relim()
#ax.set_yscale('log')
plt.ylim(ymax = y_max, ymin = y_min)


ax.set_title(chart_title)
ax.legend(loc = 'upper right')
ax.set_xlabel('Wallclock Time (seconds)')
ax.set_ylabel('Best MSE')

box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
#ax.legend(loc='top right', bbox_to_anchor=(1, 0.5))
ax.legend(loc='top right')

ax.grid()

print "saving figure as '" + figure_filename + "'"


plt.savefig(figure_filename)
plt.clf()
plt.cla()
plt.close()
