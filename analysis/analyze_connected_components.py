#!/usr/bin/env python3
# -*- coding: utf8 -*-
import networkx as nx
import numpy as np

from datasets import load_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to dataset directory")
    args = argparse.parse_args()
    print("Loading data from:", args.path)
    g, feats, labels, years = load_data(args.path)

    print("Converting graph to networkx")
    g = g.to_networkx().to_undirected()

    print("Computing connected components")
    ccs = nx.connected_components(g)

    x = np.array([len(cc) for cc in ccs])

    print('#connected components', x.size)
    print('Min max cc size in [{:d},{:d}]'.format(x.min(), x.max()))
    print('Mean: {:.4f} ({:.4f})'.format(x.mean(), x.std()))





if __name__ == "__main__":
    main()
