#!/usr/bin/env python

# Exports results from an examm training logfile to a tabulated CSV file

import os, sys
import numpy as np
import re

def get_genome_id(genome_filename):
    nums = re.findall(r'\d+', genome_filename)
    return nums[0] 

def main(argv):
    root_dir = argv[0]
    csv_out = None

    if len(argv) > 1 and argv[1] != None:
        csv_out = open(argv[1], 'w+')

    files = os.listdir(root_dir)

    # TODO switch to a map with the genome id to mae and mse (2 maps)

    mae_errors = {}
    mse_errors = {}

    os.chdir(root_dir)
    for file in files:
        f = open(os.getcwd() + '/' + file)
        test_found = False
        for line in f:
            if test_found:
                genome_id = get_genome_id(str(file))

                if "MAE" in line:
                    x = line.split(" ")
                    f_val = x[len(x) - 1]
                    mae_errors[genome_id] = float(f_val)

                elif "MSE" in line:
                    x = line.split(" ")
                    f_val = x[len(x) - 1]
                    mse_errors[genome_id] = float(f_val)

            elif "TEST ERRORS" in line:
                test_found = True

    csv_head = "GenomeID, MAE, MSE"
    if csv_out != None:
        csv_out.write(csv_head + "\n")
    else:
        print(csv_head)

    for key in mae_errors.keys():
        info = key + ", " + str(mae_errors[key]) + ", " + str(mse_errors[key])
        if csv_out != None:
            csv_out.write(info + "\n")
        else:
            print(info)



if __name__ == "__main__":
   main(sys.argv[1:])
