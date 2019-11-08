# plot worst/best fitness of each island,
# fitness file must have worst and best fitness of each island
# arguments: fit_file: fiteness file address
#            plot_identifier: name prefix for the saved plot

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd

fit_file = sys.argv[1]
plot_identifier = sys.argv[2]
N_row = 2
N_column = 5
N_check = 200
plot_dir= "./plots"

def get_ratio(n, data):
    ratio = []
    for i in range(n):
        # best/worst
        temp = np.divide(data[:, 2*i], data[:, 2*i+1]).reshape((-1, 1))
        ratio.append(temp)
    length = temp.shape[0]
    ratio = np.asarray(ratio).reshape((n, length))

    return ratio


def plot_island(ratio, x_check):
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    fig=plt.figure()
    k = 0
    for i in range(N_row):
        for j in range(N_column):
            plt.subplot2grid((N_row, N_column), (i, j))
            plt.plot(ratio[k, :])
            plt.scatter(x=x_check, y=ratio[k, x_check],c='r',marker='o')
            plt.grid(True)
            k = k+1

    fig.suptitle(plot_identifier)
    plt.savefig(plot_dir+"/worst_best_" + plot_identifier + ".png")
    plt.show()


def plot_island_individual(ratio, x_check):

    k = 0
    for i in range(N_row):
        for j in range(N_column):
            plt.figure(k)
            # plt.subplot2grid((N_row, N_column), (i, j))
            plt.plot(ratio[k, :])
            plt.scatter(x=x_check, y=ratio[k, x_check], c='r', marker='o')
            plt.grid(True)
            k = k+1
            plt.title(plot_identifier+"_island_"+str(k))
            plt.savefig(plot_dir+"/worst_best_" + plot_identifier +"_"+ str(k) + ".png") 
            plt.show()

    
def main():

    in_data = pd.read_csv(fit_file)
    fit_data = in_data.values[:, 8:]
    n = int(fit_data.shape[1]/2)
    x_domain_max = fit_data.shape[0]
    x_check = np.arange(0, x_domain_max, N_check)
    ratio = get_ratio(n, fit_data)
    print(ratio.shape)
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    # can choose to plot islands together or individually
    plot_island(ratio, x_check)
    plot_island_individual(ratio, x_check)


if __name__ == "__main__":
    main()
