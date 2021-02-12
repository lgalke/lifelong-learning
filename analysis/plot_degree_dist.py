import os
import os.path as osp
from datasets import load_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from collections import Counter
from operator import itemgetter
import dgl
from dgl.data import register_data_args, load_data as load_data_dgi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help="Path to 70company-like dataset")
    parser.add_argument('--outfile', help="Path to 70company-like dataset",
                        default='./figures/degree.png')
    parser.add_argument('--dmin', help="Custom min degree for computing power law exp, defaults to min value from data",
                        default=None, type=int)
    parser.add_argument('--remove-self-loops', help="Remove self loops",
                        default=False, action='store_true')

    args = parser.parse_args()

    try: 
        g, _, _, _ = load_data(args.path)
    except FileNotFoundError:
        print("Trying to load", args.path, "via DGL")
        g = dgl.DGLGraph(load_data_dgi(argparse.Namespace(dataset=args.path)).graph)
    print(g)
    if args.remove_self_loops:
        print("Removing self-loops")
        g = dgl.transform.remove_self_loop(g)
    degrees = np.asarray(g.in_degrees())

    degree_counts = Counter(degrees)
    x, y = zip(*degree_counts.items())

    plt.figure(1)

    plt.title("Degree distribution")

    plt.xlabel('degree')
    plt.xscale('log')
    plt.xlim(min(x), max(x))

    plt.ylabel('frequency')
    plt.yscale('log')
    plt.ylim(min(y), max(y))

    # start with straight line
    # plt.plot([min(x), max(x)], [max(y), min(y)], color='r', linestyle='-', linewidth=1.0)
    # and scatter the distribution 
    plt.scatter(x, y, marker='.')
    
    print(f"Plotting to {args.outfile}")
    plt.savefig(args.outfile)


    print("Computing power law exponent")

    if args.dmin is None:
        dmin = degrees.min()
    else:
        degrees = degrees[degrees >= args.dmin]
        dmin = args.dmin

    # N must be number of values that go into computation
    # Not total number of nodes
    n = degrees.size
    print("d_min =", dmin)
    print("N =", n)
    gamma = 1 + n / np.log(degrees / dmin).sum()
    print("Gamma = {:.4f}".format(gamma))



if __name__ == '__main__':
    main()



