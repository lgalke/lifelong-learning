import os
import argparse
from datasets import load_data
import torch

from lifelong_learning import make_lifelong_nodeclf_dataset, lifelong_nodeclf_identifier


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset")
    parser.add_argument("--t_zero", default=None, type=int, help="Last task identifier before the first evaluation task identifier")
    parser.add_argument("--history", default=0, type=int, help="History size")
    parser.add_argument("--backend", default="dgl", type=str, choices=["dgl", "geometric"])
    parser.add_argument("--basedir", help="Basedir for preprocessed dataset, else create subdirectory in input")
    parser.add_argument("--label_rate", help="Subsample the train nodes globally.", default=None, type=float)
    args = parser.parse_args()
    if args.label_rate is not None:
        assert args.label_rate > 0.0 and args.label_rate < 1.0
    graph_or_edge_index, features, labels, years = load_data(args.dataset, backend=args.backend)
    basedir = args.basedir if args.basedir else args.dataset

    outdir = os.path.join(basedir, lifelong_nodeclf_identifier(args.dataset, args.t_zero, args.history, args.backend, label_rate=args.label_rate))

    # Cast to torch tensors
    features = torch.as_tensor(features, dtype=torch.float)
    labels = torch.as_tensor(labels, dtype=torch.long)
    years = torch.as_tensor(years, dtype=torch.long)

    if args.backend == "geometric":
        dataset = make_lifelong_nodeclf_dataset(outdir,
                                                years,
                                                features,
                                                labels,
                                                edge_index=graph_or_edge_index,
                                                t_zero=args.t_zero,
                                                cumulate=args.history,
                                                subsample_train=args.label_rate)
    elif args.backend == 'dgl':
        dataset = make_lifelong_nodeclf_dataset(outdir,
                                                years,
                                                features,
                                                labels,
                                                dgl_graph=graph_or_edge_index,
                                                t_zero=args.t_zero,
                                                cumulate=args.history,
                                                subsample_train=args.label_rate)
    else:
        raise ValueError("Unknown backend")

    print(dataset)


if __name__ == '__main__':
    main()
