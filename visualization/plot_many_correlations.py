import matplotlib.pyplot as plt
import numpy as np
import math

from numpy import genfromtxt
from numpy import loadtxt
#from pylab import polyfit
#from scipy.stats import linregress
import pandas as pd
import sys

import os

if len(sys.argv) != 2:
    print("usage: python " + sys.argv[0] + " <input_directory>")
    exit(1)

def listdir_fullpath(d):
        return [os.path.join(d, f) for f in os.listdir(d)]

for file in listdir_fullpath(sys.argv[1]):
    if file.endswith(".csv"):
        csv_filename = file
        headers_filename = csv_filename.replace("_correlations.csv", "_headers.txt")
        output_filename = csv_filename.replace("_correlations.csv", "_heatmap.png")
        
        command = "python3 plot_correlation_heatmap.py \"%s\" \"%s\" \"%s\"" % (csv_filename, headers_filename, output_filename)
        print(command)
        os.system(command)

