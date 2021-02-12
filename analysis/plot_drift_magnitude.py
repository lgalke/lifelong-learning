import os.path as osp

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn

from drift_magnitude import drift_magnitude_per_time, VALID_METRICS


DATASETS = [("pharmabio", 1999), ("dblp-easy", 2004), ("dblp-hard", 2004)]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('basedir', help="Path to dataset basedir")
    parser.add_argument('-m', '--metric', default='total_variation', choices=VALID_METRICS, type=str, help="Sliding window size for previous labels")
    parser.add_argument('--history', default=None, type=int, help="Sliding window size for previous labels")
    parser.add_argument('--cumulative', default=False, action='store_true', help="Apply history also to 'right-side'")
    parser.add_argument('--save_plot', default=None, type=str, help="Path to save plot")
    parser.add_argument('--info', default=False, action='store_true', help="Print some more info")
    parser.add_argument('--fontscale', default=1.0, type=float,  help="Fontscale")
    args = parser.parse_args()
    verbose = args.info
    seaborn.set(font_scale=args.fontscale)


    data = {
            "dataset": [],
            "time": [],
            "drift": []
            }

    for dataset, t_start in DATASETS:
        print(f"Processing {dataset} with t_start = {t_start}")
        time = np.load(osp.join(args.basedir, dataset, "t.npy"))
        labels = np.load(osp.join(args.basedir, dataset, "y.npy"))
        t, dm = drift_magnitude_per_time(time, labels, t_start=t_start, verbose=verbose,
                    history=args.history, cumulative=args.cumulative, metric=args.metric)

        data["dataset"].extend([dataset] * len(t))
        data["time"].extend(list(t))
        data["drift"].extend(list(dm))

    g = seaborn.relplot(x="time", y="drift", style="dataset", hue="dataset", markers=True, kind="line", height=4, aspect=1.414752116, data=data, palette="colorblind")
    g.set(xticks=np.unique(data["time"])[::5], ylabel="drift magnitude [0,1]")
    if args.save_plot:
        plt.savefig(args.save_plot)
    else:
        plt.show()

if __name__ == '__main__':
    main()
