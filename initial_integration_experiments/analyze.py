import pandas

import numpy as np

import matplotlib.pyplot as plt

fig, subplts = plt.subplots(6, 1)

bprange = [1, 2, 4, 8, 16, 32]
plts = {k:v for k, v in zip(bprange, subplts)}
print(plts)
base = plts[bprange[0]]

for k, v in plts.items():
    v.set_title(f"{k} BPI")
    if k == bprange[0]:
        continue
    v.sharey(base)
    v.sharex(base)

results = {}
for ci in [64, 128, 256, 512]:
    results[ci] = {}
    for bpe in bprange:
        results[ci][bpe] = {}
        for k in [1]:
            x = []
            results[ci][bpe][k] = x

            for fold in range(8):
                f = pandas.read_csv(f"initial_integration_experiments/results/v3/{ci}/{bpe}/{k}/{fold}/fitness_log.csv")
                results[ci][bpe][k].append(f)


            enabled_nodes = []
            enabled_edges = []
            enabled_rec_edges = []

            bpi_columns = []
            mse_columns = []

            minlen = 100000000

            for f in x:
                bpi_columns.append(f[' Total BP Epochs'].to_numpy())
                mse_columns.append(f[' Best Val. MSE'].to_numpy())
                enabled_nodes.append(f[' Enabled Nodes'].to_numpy())
                enabled_edges.append(f[' Enabled Edges'].to_numpy())
                enabled_rec_edges.append(f[' Enabled Rec. Edges'].to_numpy())

                minlen = min(minlen, len(bpi_columns[-1]))

            enabled_nodes = list(map(lambda x: x[:minlen], enabled_nodes))
            enabled_edges = list(map(lambda x: x[:minlen], enabled_edges))
            enabled_rec_edges = list(map(lambda x: x[:minlen], enabled_rec_edges))
            bpi_columns = list(map(lambda x: x[:minlen], bpi_columns))
            mse_columns = list(map(lambda x: x[:minlen], mse_columns))


            nodesmean = np.mean(np.array(enabled_nodes), axis=0)
            edgesmean = np.mean(np.array(enabled_edges), axis=0)
            redgesmean = np.mean(np.array(enabled_rec_edges), axis=0)
            print(f"Nodes at end mean: {nodesmean[-1]}")
            print(f"edges at end mean: {edgesmean[-1]}")
            print(f"redges at end mean: {redgesmean[-1]}")


            bpimean = np.mean(np.array(bpi_columns), axis=0)
            msemean = np.mean(np.array(mse_columns), axis=0)
            msestd = np.std(np.array(mse_columns), axis=0)

            g = plts[bpe].plot(bpimean, msemean, label=f"ci={ci}")[0]
            plts[bpe].fill_between(bpimean, msemean - msestd, msemean + msestd,
                alpha=0.2, edgecolor=g.get_color(), facecolor=g.get_color(), linewidth=0)

for k, v in plts.items():
    v.set_title(f"{k} BPI")
    v.legend(fontsize=12, loc="upper right")

plt.show()
