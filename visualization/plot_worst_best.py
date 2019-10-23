# plot worst/best fitness of each island, 
# fitness file must have worst and best fitness of each island
# only one argument: fitness file address
#
# Zimeng Lyu 

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd

fit_file = sys.argv[1]

N_row=2
N_column=5
def get_ratio(n, data):
    ratio = []
    for i in range(n):
        # best/worst
        temp = np.divide(data[:, 2*i], data[:, 2*i+1]).reshape((-1,1))
        ratio.append(temp)
    length = temp.shape[0]
    ratio=np.asarray(ratio).reshape((n,length))

    return ratio

def plot_island(ratio):
    plt.figure()
    k = 0
    for i in range(N_row):
        for j in range(N_column):
            plt.subplot2grid((2, 5), (i, j))
            plt.plot(ratio[k, :])
            plt.grid(True)
            plt.title('island_'+str(k))
            k = k+1
    # plt.savefig("./worst_best.png")
    plt.show()

    # plt.savefig("./worst_best.png")
def main():

    in_data=pd.read_csv(fit_file)
    fit_data = in_data.values[:, 8:]
    # print(fit_data.shape)
    n=int(fit_data.shape[1]/2)
    ratio=get_ratio(n, fit_data)
    print(ratio.shape)
    plot_island(ratio)


if __name__ == "__main__":
    main()
