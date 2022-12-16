import os
import os.path as osp
from matplotlib import pyplot as plt

plt.rcParams.update({"font.size": 16})

import numpy as np
import pandas as pd
import seaborn as sns
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to 70company-like dataset")
    parser.add_argument(
        "--outfile",
        help="Path to 70company-like dataset",
        default="./figures/current-output-debug.png",
    )

    args = parser.parse_args()
    data = pd.DataFrame(
        {
            "year": np.load(osp.join(args.path, "t.npy")),
            "label": np.load(osp.join(args.path, "y.npy")),
        }
    )

    print("Label value counts:\n", data.label.value_counts())

    plt.figure(1)
    theplot = sns.countplot(
        x="label", data=data, log=False, order=data["label"].value_counts().index
    )
    print(f"Plotting to {args.outfile}")

    for idx, label in enumerate(theplot.xaxis.get_ticklabels()):
        #        if idx % 5 != 0:
        label.set_visible(False)

    plt.gca().legend().remove()
    plt.tight_layout()
    plt.savefig(args.outfile)


if __name__ == "__main__":
    main()
