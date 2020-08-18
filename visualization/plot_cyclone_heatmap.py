<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#example usage:
#python3 plot_cyclone_heatmap.py ../build/heatmap_output.csv ../build/cyclone_heatmap.png

=======
>>>>>>> a86376b73f6b8e478b03b94e644c15ba49fe7e5d
=======
>>>>>>> a86376b73f6b8e478b03b94e644c15ba49fe7e5d
=======
>>>>>>> a86376b73f6b8e478b03b94e644c15ba49fe7e5d
import matplotlib.pyplot as plt
import numpy as np
import math

from numpy import genfromtxt
from numpy import loadtxt
#from pylab import polyfit
#from scipy.stats import linregress
import pandas as pd
import sys

if len(sys.argv) != 3:
    print("usage: python " + sys.argv[0] + " <input_file> <output_file>")
    exit(1)

for arg in sys.argv:
        print(arg)


input_file = sys.argv[1]
output_file = sys.argv[2]

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD

=======
>>>>>>> a86376b73f6b8e478b03b94e644c15ba49fe7e5d
=======
>>>>>>> a86376b73f6b8e478b03b94e644c15ba49fe7e5d
=======
>>>>>>> a86376b73f6b8e478b03b94e644c15ba49fe7e5d
headers = []
for cyclone in range(1,13):
    headers.append(str(cyclone));

print(headers)

data = genfromtxt(input_file, delimiter=',')
print(data)

fig, ax = plt.subplots()
heatmap = plt.pcolor(data, cmap = plt.cm.Blues, alpha = 0.8)
#heatmap = plt.pcolor(data, cmap='RdBu', alpha = 0.8, vmin=-0.5, vmax=0.5)
plt.colorbar()

ax.set_xticks(np.arange(len(headers)))
ax.set_yticks(np.arange(len(headers)))
# ... and label them with the respective list entries
ax.set_xticklabels(headers, fontsize=10)
ax.set_yticklabels(headers, fontsize=10)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

#plt.xticks([x-0.5 for x in list(range(1,len(data)+1))], range(1,len(data)+1))
#plt.yticks([x-0.5 for x in list(range(1,len(data)+1))], range(1,len(data)+1))

plt.xticks(np.arange(len(headers))+0.5, headers, rotation=45)
plt.yticks(np.arange(len(headers))+0.5, headers)

plt.ylabel('Training Cyclone', fontsize=14)
plt.xlabel('Testing Cyclone', fontsize=14)



#ax.set_frame_on(False)
#ax.grid(False)
#ax = plt.gca()

plt.tight_layout()


plt.savefig(output_file)
