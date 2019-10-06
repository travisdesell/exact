'''
usage examples:
    python plot_multi_time_series.py ~/Dropbox/1537\ MTI-RIT/2019_degruyter_results/net_plant_heat_rate_predictions.csv "Net Plant Heat Rate" "Hours" ~/Dropbox/1537\ MTI-RIT/2019_degruyter_results/net_plant_heat_rate_multi.png 1 2 4 8
    python plot_multi_time_series.py ~/Dropbox/1537\ MTI-RIT/2019_degruyter_results/nose_gas_temperature_predictons.csv "Nose Gas Temperature" "Hours" ~/Dropbox/1537\ MTI-RIT/2019_degruyter_results/nose_gas_temperature_multi.png 1 2 4 8
    python plot_multi_time_series.py ~/Dropbox/1537\ MTI-RIT/2019_degruyter_results/flame_intensity_extra_3.csv "Flame Intensity (Plant ~ Fuel)" "Minutes" ~/Dropbox/1537\ MTI-RIT/2019_degruyter_results/flame_intensity_extra_3.png 1 15 30 60 120 240 480
    python plot_multi_time_series.py ~/Dropbox/1537\ MTI-RIT/2019_degruyter_results/flame_intensity_plant_fuel_3.csv "Flame Intensity (Plant + Fuel)" "Minutes" ~/Dropbox/1537\ MTI-RIT/2019_degruyter_results/flame_intensity_plant_fuel_3.png 1 15 30 60 120 240 480
    python plot_multi_time_series.py ~/Dropbox/1537\ MTI-RIT/2019_degruyter_results/flame_intensity_plant_3.csv "Flame Intensity (Plant Only)" "Minutes" ~/Dropbox/1537\ MTI-RIT/2019_degruyter_results/flame_intensity_plant_3.png 1 15 30 60 120 240 480

'''

import sys
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

if len(sys.argv) < 5:
    print "invalid arguments, usage:"
    print "python " + sys.argv[0] + " <csv filename> <parameter name> <xlabel name> <output filename> <time offsets 1..n>"

    exit(1)

csv_filename = sys.argv[1]
parameter_name = sys.argv[2]
xlabel_name = sys.argv[3]
output_filename = sys.argv[4]
offsets = sys.argv[5:]

print "offsets: " + str(offsets)

#values = np.loadtxt(csv_filename, delimiter=",")
values = genfromtxt(csv_filename, delimiter=",")

rows = len(values)
cols = len(values[0])

print "rows x cols: " + str(cols) + " x " + str(rows)

plt.figure(1, figsize=(30,5))

for col in range(0, cols):
    print "col: " + str(col)

    print values[:,col]


    if col == 0:
        plt.plot(values[:,col], label="Actual " + parameter_name, linewidth=0.25)
    else:
        plt.plot(values[:,col], label="Predicted " + parameter_name + " Offset " + offsets[col - 1], linewidth=0.25)

plt.legend(loc='lower right')

plt.title(parameter_name + " Predictions")
plt.ylabel("Normalized " + parameter_name)
plt.xlabel(xlabel_name)
plt.grid(True)
plt.savefig(output_filename, bbox_inches='tight')

