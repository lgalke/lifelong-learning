import os
import os.path as osp

import numpy as np
import pandas as pd
import argparse

import numpy as np
import scipy.stats as stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help="Path to 70company-like dataset")

    args = parser.parse_args()
    data = pd.DataFrame(
        {
          'year': np.load(osp.join(args.path, 't.npy')),
         'label': np.load(osp.join(args.path, 'y.npy'))
        }
    )
    value_counts = data.label.value_counts()
    # print("Label value counts:\n", value_counts)
    print("Entropy (base e):", stats.entropy(value_counts))
    h = stats.entropy(value_counts, base=2)
    print("Entropy (base 2):", h)
    num_classes = len(data.label.unique())
    print(f"Entropy (base {num_classes}):", stats.entropy(value_counts, base=num_classes))

    num_classes = len(data.label.unique())
    h_max = stats.entropy(np.repeat(1, num_classes), base=2)
    print("H_max =", h_max)
    print("Entropy (base 2) normalized by H_max", h / h_max)


if __name__ == '__main__':
    main()
