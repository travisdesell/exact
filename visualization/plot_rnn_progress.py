import csv
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
#from pylab import polyfit
from scipy.stats import linregress
import pandas as pd
import sys
from cycler import cycler


input_file = sys.argv[1]
output_file = sys.argv[2]

f = open(input_file, 'rb')
#reader = csv.reader(f)
#headers = next(reader)

#print(headers)

v1 = genfromtxt(input_file)
#v2 = genfromtxt('migration_ring_salsa_10_320.txt', delimiter=' ')
#print v1

#print v1

#print "first column of v1:\n"
t = [row[0] for row in v1]
mse = [row[1] for row in v1]
v_mse = [row[2] for row in v1]
bv_mse = [row[3] for row in v1]
#print("t:\n")
#print(t)
#print("mse:\n")
#print(mse)
#print("norm:\n")
#print(norm)


fig, ax1 = plt.subplots(1)
fig.set_size_inches(20.0, 5.0)

ax1.grid()
ax1.set_title('RNN Progress')
ax1.set_xlabel('Epoch')


ax1.set_yscale('log')
ax1.set_ylabel('Training MSE', color='b')
ax1.tick_params('y', colors='b')
ax1.plot(t, mse, lw=0.25, label='Training MSE', color='blue')
ax1.plot(t, v_mse, lw=0.25, label='Validation MSE', color='green')
ax1.plot(t, bv_mse, lw=0.25, label='Best Validation MSE', color='red')
ax1.set_ylim(0.001, 10)


'''
ax2 = ax1.twinx()
ax2.set_yscale('log')
ax2.set_ylabel('Validation MSE', color='g')
ax2.tick_params('y', colors='g')
ax2.plot(t, v_mse, lw=0.25, label='Validation MSE', color='green')
#ax2.set_ylim(0.0000001, 1)

ax3 = ax1.twinx()
ax2.set_yscale('log')
ax3.set_ylabel('Best Validation MSE', color='r')
ax2.tick_params('y', colors='r')
ax2.plot(t, bv_mse, lw=0.25, label='Best Validation MSE', color='red')
#ax2.set_ylim(0.0000001, 1)
'''


ax1.legend(loc = 'upper right')

plt.savefig(output_file, dpi=100)
plt.clf()
plt.cla()
plt.close()

