#!/usr/bin/env python3

import sys

in_file = open(sys.argv[1], "r")
brk = " "
bv_mse = "bv_mse"

for line in in_file:
    if "iteration" in line:
        strs = line.split(" ")
        for j in range(0, len(strs)):
            str = strs[j]
            if bv_mse in str:
                c = strs[j+1]

print(c[0:len(c) - 1])
in_file.close()
