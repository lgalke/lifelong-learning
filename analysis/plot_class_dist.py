import os
import os.path as osp
from datasets import load_70companies_dataframe
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help="Path to 70company-like dataset")
    parser.add_argument('--outfile', help="Path to 70company-like dataset",
                        default='./figures/years.png')

    args = parser.parse_args()
    data = pd.DataFrame(
        {
          'year': np.load(osp.join(args.path, 't.npy')),
         'label': np.load(osp.join(args.path, 'y.npy'))
        }
    )


    print("Label value counts:\n", data.label.value_counts())

    plt.figure(1)
    sns.countplot(x="label", data=data, log=False, order=data["label"].value_counts().index)
    print(f"Plotting to {args.outfile}")
    plt.gca().legend().remove()
    plt.savefig(args.outfile)

if __name__ == '__main__':
    main()




