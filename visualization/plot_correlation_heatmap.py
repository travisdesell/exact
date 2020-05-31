import matplotlib.pyplot as plt
import numpy as np
import math

from numpy import genfromtxt
from numpy import loadtxt
#from pylab import polyfit
#from scipy.stats import linregress
import pandas as pd
import sys

if len(sys.argv) != 4:
    print("usage: python " + sys.argv[0] + " <input_file> <header_file> <output_file>")
    exit(1)

for arg in sys.argv:
        print(arg)


input_file = sys.argv[1]
header_file = sys.argv[2]
output_file = sys.argv[3]


headers_file = open(header_file, "r")
headers = headers_file.read().split("\n")

print(headers)

data = genfromtxt(input_file, delimiter=',')
print(data)

fig, ax = plt.subplots()
#heatmap = plt.pcolor(data, cmap = plt.cm.Blues, alpha = 0.8)
heatmap = plt.pcolor(data, cmap='RdBu', alpha = 0.8, vmin=-1, vmax=1)
plt.colorbar()


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize


ax.set_xticks(np.arange(data.shape[1])      , minor = False)
ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor = False)

ax.invert_yaxis()
ax.xaxis.tick_top()

fig.set_size_inches(40,10)
#fig.set_size_inches(90,90)
ax.set_xticklabels(range(0,data.shape[1]), minor = False, fontsize = 4)
ax.set_yticklabels(headers, minor = False, fontsize = 8)


ax.set_frame_on(False)
ax.grid(False)
ax = plt.gca()

plt.xticks(rotation=90)

for t in ax.xaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False

for t in ax.yaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False


plt.savefig(output_file)
