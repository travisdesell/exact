import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

if len(sys.argv) != 7:
    print "invalid arguments, usage:"
    print "python " + sys.argv[0] + " <csv filename> <parameter name> <xlabel name> <parameter_column> <prediction_column> <output filename>"

    exit(1)

csv_filename = sys.argv[1]
parameter_name = sys.argv[2]
xlabel_name = sys.argv[3]
parameter_column = int(sys.argv[4])
prediction_column = int(sys.argv[5])
output_filename = sys.argv[6]


values = np.loadtxt(csv_filename, delimiter=",")

print "actual " + parameter_name + " (column " + str(parameter_column) + "):"
print values[:,parameter_column]
print "predicted " + parameter_name + " (column " + str(prediction_column) + "):"
print values[:,prediction_column]

plt.figure(1, figsize=(30,5))

plt.plot(values[:,parameter_column], label="Actual " + parameter_name, linewidth=0.25)
plt.plot(values[:,prediction_column], label="Predicted " + parameter_name, linewidth=0.25)
plt.legend(loc='lower right')

plt.title(parameter_name + "Predictions")
plt.ylabel("Normalized " + parameter_name)
plt.xlabel(xlabel_name)
plt.grid(True)
plt.savefig(output_filename)

