#!/usr/bin/env python3

import sys

in_file = open(sys.argv[1], "r")
brk = int(sys.argv[2])


c = 0
for line in in_file:
    strs = line.split(" ")
    time = strs[4];

    times = strs[4].split("m")

    seconds = float(times[1][0:len(times[1]) - 2])
    mins = float(times[0])

    d_sec = (mins * 60) + seconds

    print(d_sec)

    if c > 0 and (c + 1) % brk == 0:
        print()

    c = c + 1

in_file.close()
