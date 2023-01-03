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
        default="./new-figures/debug.png",
    )

    args = parser.parse_args()
    data = pd.DataFrame(
        {
            "year": np.load(osp.join(args.path, "t.npy")),
            "label": np.load(osp.join(args.path, "y.npy")),
        }
    )

    print(data.year.describe())
    # print("Year value counts:\n", data.year.value_counts())

    plt.figure(1)
    # sns.countplot(y="year", hue="label", data=data, log=True)
    theplot = sns.countplot(y="year", data=data, log=True)
    for label in theplot.yaxis.get_ticklabels():
        actual_label = label.get_text()
        if not (actual_label.endswith("0") or actual_label.endswith("5")):
            label.set_visible(False)
    print(f"Plotting to {args.outfile}")
    plt.gca().legend().remove()
    plt.tight_layout()
    plt.savefig(args.outfile)


if __name__ == "__main__":
    main()
