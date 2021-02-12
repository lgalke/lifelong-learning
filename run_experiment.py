#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import gc

import numpy as np
import pandas as pd
import dgl
import torch_geometric as tg
import torch
# import torch.nn as nn
import torch.nn.functional as F

# Models
from sklearn.metrics import f1_score

from models import GraphSAGE
from models import GAT
from models import MLP
from models import MostFrequentClass
from models import JKNet
from models.sgnet import SGNet

from models.graphsaint import train_saint, evaluate_saint
from models import geometric as geo
from models.node2vec import add_node2vec_args, train_node2vec, evaluate_node2vec


# EvolveGCN
# from models.evolvegcn.egcn_o import EGCN
# from models.evolvegcn.models import Classifier
# import models.evolvegcn.utils as egcn_utils

from datasets import load_data


def appendDFToCSV_void(df, csvFilePath, sep=","):
    """ Safe appending of a pandas df to csv file
    Source: https://stackoverflow.com/questions/17134942/pandas-dataframe-output-end-of-csv
    """
    if not os.path.isfile(csvFilePath):
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep)
    elif len(df.columns) != len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns):
        raise Exception(
            "Columns do not match!! Dataframe has " + str(len(df.columns)) + " columns. CSV file has " + str(
                len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns)) + " columns.")
    elif not (df.columns == pd.read_csv(csvFilePath, nrows=1, sep=sep).columns).all():
        raise Exception("Columns and column order of dataframe and csv file do not match!!")
    else:
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep, header=False)


def compute_weights(ts, exponential_decay, initial_quantity=1.0, normalize=True):
    ts = torch.as_tensor(ts)
    delta_t = ts.max() - ts
    values = initial_quantity * torch.exp(- exponential_decay * delta_t)
    if normalize:
        # When normalizing, the initial_quantity is irrelevant
        values = values / values.sum()
    return values


def train(model, optimizer, g, feats, labels, mask=None, epochs=1, weights=None,
          backend='dgl'):
    model.train()
    reduction = 'none' if weights is not None else 'mean'

    if hasattr(model, '__reset_cache__'):
        print("Resetting Model Cache")
        model.__reset_cache__()

    for epoch in range(epochs):
        inputs = (g, feats) if backend == 'dgl' else (feats, g)
        logits = model(*inputs)

        if mask is not None:
            loss = F.cross_entropy(logits[mask], labels[mask], reduction=reduction)
        else:
            loss = F.cross_entropy(logits, labels, reduction=reduction)

        if weights is not None:
            loss = (loss * weights).sum()

        # Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epoch {:d} | Loss: {:.4f}".format(epoch + 1, loss.detach().item()))


def evaluate(model, g, feats, labels, mask=None, compute_loss=True,
             backend='dgl'):
    model.eval()

    if hasattr(model, '__reset_cache__'):
        print("Resetting Model Cache")
        model.__reset_cache__()

    with torch.no_grad():
        inputs = (g, feats) if backend == 'dgl' else (feats, g)
        logits = model(*inputs)

        if mask is not None:
            logits = logits[mask]
            labels = labels[mask]

        if compute_loss:
            loss = F.cross_entropy(logits, labels).item()
        else:
            loss = None

        if isinstance(logits, np.ndarray):
            logits = torch.FloatTensor(logits)
        __max_vals, max_indices = torch.max(logits.detach(), 1)
        acc = (max_indices == labels).sum().float() / labels.size(0)
        f1 = f1_score(labels.cpu(), max_indices.cpu(), average="macro")

    return acc.item(), f1, loss


def build_model(args, in_feats, n_hidden, n_classes, device, n_layers=1, backend='geometric',
                edge_index=None, num_nodes=None):
    if args.model == 'graphsaint':
        assert backend == 'geometric'
        model_spec = args.variant
    else:
        model_spec = args.model

    if backend == 'geometric':
        print("Using Geometric Backend")
        if model_spec == 'gs-mean':
            model = geo.GraphSAGE(in_feats, n_hidden, n_classes, n_layers, F.relu, args.dropout).to(device)
        elif model_spec == "gcn":
            model = geo.GCN(in_feats, n_hidden, n_classes, n_layers, F.relu, args.dropout).to(device)
        elif model_spec == "gat":
            print("Warning, GAT doesn't respect n_layers")
            heads = [8, args.gat_out_heads]  # Fixed head config
            n_hidden_per_head = int(n_hidden / heads[0])
            model = geo.GAT(in_feats, n_hidden_per_head, n_classes, F.relu, args.dropout, 0.6, heads).to(device)
        elif model_spec == "mlp":
            model = geo.MLP(in_feats, n_hidden, n_classes, n_layers, F.relu, args.dropout).to(device)
        elif model_spec == 'jknet-sageconv':
            # Geometric JKNEt with SAGECOnv
            model = JKNet(tg.nn.SAGEConv, in_feats, n_hidden, n_classes, n_layers, F.relu, args.dropout,
                    mode="cat", conv_kwargs={"normalize": False}, backend="geometric").to(device)
        elif model_spec == 'jknet-graphconv':
            model = JKNet(tg.nn.GraphConv, in_feats, n_hidden, n_classes, n_layers, F.relu, args.dropout,
                    mode="cat", conv_kwargs={"aggr": "mean"}, backend="geometric").to(device)
        elif model_spec == "sgnet":
            model = geo.SGNet(in_channels=in_feats, out_channels=n_classes, K=n_layers, cached=True).to(device)
        else:
            raise NotImplementedError(f"Unknown model spec 'f{model_spec} for backend {backend}")
    elif backend == 'dgl': # DGL models
        if model_spec == 'gs-mean':
            model = GraphSAGE(in_feats, n_hidden, n_classes,
                              n_layers, F.relu, args.dropout,
                              'mean').to(device)
        elif model_spec == 'mlp':
            model = MLP(in_feats, n_hidden, n_classes,
                        n_layers, F.relu, args.dropout).to(device)
        elif model_spec == 'mostfrequent':
            model = MostFrequentClass()
        elif model_spec == 'gat':
            print("Warning, GAT doesn't respect n_layers")
            heads = [8, args.gat_out_heads]  # Fixed head config
            # Div num_hidden by heads for same capacity
            n_hidden_per_head = int(n_hidden / heads[0])
            assert n_hidden_per_head * heads[0] == n_hidden, f"{n_hidden} not divisible by {heads[0]}"
            model = GAT(1, in_feats, n_hidden_per_head, n_classes,
                        heads, F.elu, 0.6, 0.6, 0.2, False).to(device)
        elif model_spec == 'node2vec':
            assert edge_index is not None
            model = tg.nn.Node2Vec(
                edge_index,
                n_hidden,
                args.n2v_walk_length,
                args.n2v_context_size,
                walks_per_node=args.n2v_walks_per_node,
                p=args.n2v_p,
                q=args.n2v_q,
                num_negative_samples=args.n2v_num_negative_samples,
                num_nodes=num_nodes,
                sparse=True
            )
        elif model_spec == 'jknet-sageconv':
            # DGL JKNet
            model = JKNet(dgl.nn.pytorch.SAGEConv,
                    in_feats, n_hidden, n_classes, n_layers, F.relu, args.dropout,
                    mode="cat", conv_args=["mean"], backend='dgl').to(device)
        elif model_spec == 'sgnet':
            model = SGNet(in_feats, n_classes, k=n_layers, cached=True, bias=True, norm=None).to(device)
        else:
            raise NotImplementedError(f"Unknown model spec 'f{model_spec} for backend {backend}")
    else:
        raise NotImplementedError(f"Unknown backend: {backend}")

    return model


def prepare_data_for_year(graph, features, labels, years, current_year, history, exclude_class=None,
                          device=None, backend='dgl', num_hops=None):
    print("Preparing data for year", current_year)
    # Prepare subgraph
    subg_node_mask = ((years <= current_year) & (years >= (current_year - history)))
    subg_nodes = torch.arange(features.size(0))[subg_node_mask]

    subg_num_nodes = subg_nodes.size(0)

    if backend == 'dgl':
        print("Creating dgl subgraph")
        subg = dgl.node_subgraph(graph, subg_nodes)
        print("Subgraph type:", type(subg))
        subg.set_n_initializer(dgl.init.zero_initializer)
    elif backend == 'geometric':
        print("Creating geometric subgraph")
        subg, __edge_attr = tg.utils.subgraph(subg_node_mask,
                                              graph, relabel_nodes=True,
                                              num_nodes=subg_num_nodes)

    else:
        raise ValueError("Unkown backend: " + backend)

    subg_features = features[subg_nodes]
    subg_labels = labels[subg_nodes]
    subg_years = years[subg_nodes]

    # Prepare masks wrt *subgraph*
    # train_nid = torch.arange(subg_num_nodes)[subg_years < current_year]
    # test_nid = torch.arange(subg_num_nodes)[subg_years == current_year]
    # print("[{}] #Training: {}".format(current_year, train_nid.size(0)))
    # print("[{}] #Test    : {}".format(current_year, test_nid.size(0)))

    train_nid = subg_years < current_year
    test_nid = subg_years == current_year
    print("[{}] #Training: {}".format(current_year, train_nid.sum()))
    print("[{}] #Test    : {}".format(current_year, test_nid.sum()))


    if device is not None:
        subg = subg.to(device)
        subg_features = subg_features.to(device)
        subg_labels = subg_labels.to(device)
        # train_nid = train_nid.to(device)
        # test_nid = test_nid.to(device)
    return subg, subg_features, subg_labels, subg_years, train_nid, test_nid


RESULT_COLS = ['dataset',
               'seed',
               'backend',
               'model',
               'variant',
               'n_params',
               'n_hidden',
               'n_layers',
               'dropout',
               'history',
               'sampling',
               'batch_size',
               'saint_coverage',
               'limited_pretraining',
               'initial_epochs',
               'initial_lr',
               'initial_wd',
               'annual_epochs',
               'annual_lr',
               'annual_wd',
               'start',
               'decay',
               'year',
               'epoch',
               'f1_macro',
               'accuracy']


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    has_parameters = args.model not in ['most_frequent']
    backend = args.backend

    print("Using backend:", backend)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    if args.model == 'mostfrequent':
        # Makes no sense to put things on GPU when using simple most frequent classifier
        device = torch.device("cpu")

    graph, features, labels, years = load_data(args.data_path, backend=backend)
    if backend == 'geometric':
        graph = graph
        features = features.float()
        labels = labels
        years = years
    else:
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)
        years = torch.LongTensor(years)


    num_nodes = features.shape[0]
    num_edges = graph.number_of_edges() if backend == 'dgl' else graph.size(1)

    print("Min year:", years.min())
    print("Max year:", years.max())
    print("Number of nodes:", num_nodes)
    print("Number of edges:", num_edges)

    # try:
    #     features = torch.FloatTensor(features.float())
    # except AttributeError:
    #     features = torch.FloatTensor(features)

    # labels = torch.LongTensor(labels)
    # years = torch.LongTensor(years)
    n_classes = torch.unique(labels).size(0)


    in_feats = features.shape[1]
    n_layers = args.n_layers
    n_hidden = args.n_hidden

    model = build_model(args, in_feats, n_hidden, n_classes, device,
                        n_layers=args.n_layers, backend=backend,
                        edge_index=graph, num_nodes=num_nodes)

    print(model)
    num_params = sum(np.product(p.size()) for p in model.parameters())
    print("#params:", num_params)
    if has_parameters:
        if args.model == 'node2vec':
            # Use SparseAdam for node2vec to speed things up
            optimizer = torch.optim.SparseAdam(model.parameters(),
                                               lr=args.lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(),
                                        lr=args.lr,
                                        weight_decay=args.weight_decay)

    results_df = pd.DataFrame(columns=RESULT_COLS)

    def attach_score(df, year, epoch, accuracy, f1):
        """ Partial """
        return df.append(
            pd.DataFrame(
                [[args.dataset,
                  args.seed,
                  backend,
                  args.model,
                  args.variant,
                  num_params,
                  args.n_hidden,
                  args.n_layers,
                  args.dropout,
                  args.history,
                  args.sampling,
                  args.batch_size,
                  args.saint_coverage,
                  args.limited_pretraining,
                  args.initial_epochs,
                  args.lr,
                  args.weight_decay,
                  args.annual_epochs,
                  args.lr * args.rescale_lr,
                  args.weight_decay * args.rescale_wd,
                  args.start,
                  args.decay,
                  year,
                  epoch,
                  f1,
                  accuracy]],
                columns=RESULT_COLS),
            ignore_index=True)

    known_classes = set()

    if not args.limited_pretraining and not args.start == 'cold' and args.initial_epochs > 0:
        # With 'limited pretraining' we do the initial epochs on the first wnidow
        # With cold start, no pretraining is needed
        # When initial epochs are 0, no pretraining is needed either
        # For current experiments, we have set initial_epochs = 0
        # Exclusively the static model of experiment 1 uses this pretraining
        data = prepare_data_for_year(graph,
                                     features,
                                     labels,
                                     years,
                                     args.pretrain_until,
                                     10000,
                                     device=device,
                                     backend=backend)
        subg, subg_features, subg_labels, subg_years, train_nid, test_nid = data
        # Use all nodes of initial subgraph for training
        print("Using data until", args.pretrain_until, "for training")
        print("Selecting", subg_features.size(0), "of", features.size(0), "papers for initial training.")

        train_nids = torch.cat([train_nid, test_nid])  # use all nodes in subg for initial pre-training
        if args.model == 'mostfrequent':
            model.fit(None, subg_labels)
        elif args.model == 'node2vec':
            train_node2vec(model, optimizer, epochs=epochs,
                           batch_size=args.n2v_batch_size,
                           shuffle=True,
                           num_workers=args.n2v_num_workers)
            acc = evaluate_node2vec(model, subg_labels, train_nid, test_nid)
        elif args.model == "graphsaint":
            raise NotImplementedError("Legacy code, needs recheck")
            train_saint(model, optimizer, subg, subg_features, subg_labels,
                        epochs=args.initial_epochs,
                        n_jobs=args.saint_njobs)
            acc, f1, _ = evaluate_saint(model, subg, subg_features, subg_labels, mask=None,
                                        backend=backend)
            print(f"** Train Accuracy {acc:.4f} **")
        else:
            print("Subg labels", subg_labels.size())
            train(model, optimizer, subg, subg_features, subg_labels,
                  mask=train_nid,
                  epochs=args.initial_epochs, backend=backend)
            acc, f1, _ = evaluate(model, subg, subg_features, subg_labels, mask=None,
                                  backend=backend)
            print(f"** Train Accuracy {acc:.4f} **")

        known_classes |= set(subg_labels.cpu().numpy())
        print("Known classes:", known_classes)

    remaining_years = torch.unique(years[years > args.pretrain_until], sorted=True)

    for t, current_year in enumerate(remaining_years):

        # print(f"allocated: {torch.cuda.memory_allocated() / 1000000000} GB")
        # Get the current subgraph

        if args.model in ['graphsaint']:
            print("///////////////////")
            print("//// inductive ////")
            print("///////////////////")
            # Train completely on Task t-1
            year_cutoff = current_year - 1
            globals_device = torch.device("cpu")
            inductive = True
        else:
            print("//////////////////////")
            print("//// transductive ////")
            print("//////////////////////")
            year_cutoff = current_year
            globals_device = device
            inductive = False

        data = prepare_data_for_year(graph,
                                     features,
                                     labels,
                                     years,
                                     year_cutoff,
                                     args.history,
                                     device=globals_device,
                                     backend=backend)
        subg, subg_features, subg_labels, subg_years, train_nid, test_nid = data

        if args.decay is not None:
            # Use decay factor to weight the loss function based on time steps t
            weights = compute_weights(years[train_nid], args.decay, normalize=True).to(device)
        else:
            weights = None

        if args.history == 0:
            # No history means no uptraining at all!!!
            # Unused. For the static model (Exp. 1) we give a history frame but do no uptraining instead.
            epochs = 0
        elif args.limited_pretraining and t == 0:
            # Do the pretraining on the first history window
            # with `initial_epochs` instead of `annual_epochs`
            epochs = args.initial_epochs
        else:
            epochs = args.annual_epochs

        if inductive:
            # Task is used completely for training
            new_classes = set(subg_labels.cpu().numpy()) - known_classes
        else:
            new_classes = set(subg_labels[train_nid].cpu().numpy()) - known_classes
        print(f"New classes at time {current_year}:", new_classes)

        if args.start == 'legacy-cold':
            # Brute force re-init of model
            del model
            model = build_model(args, in_feats, n_hidden, n_classes, device, n_layers=args.n_layers,
                                edge_index=subg, num_nodes=subg_features.size(0), backend=backend)
        elif args.start == 'cold' or (args.start == 'hybrid' and new_classes):
            # NEW version, equivalent to legacy-cold, but more efficient
            model.reset_parameters()
        elif args.start == 'legacy-warm' or (args.start == 'hybrid' and not new_classes):
            # Legacy warm start: just keep old params as is
            # differs from new warm variant on unseen classes with cat. CE loss
            pass
        elif args.start == 'warm':
            # Skip for first task (does not make sense and makes problem for SGNET)
            if t > 0 and new_classes and has_parameters:
                print("~~~~~~ Doing partial warm reinit ~~~~~~")
                # If there are new classes:
                # 1) Save parameters of final layer
                # 2) Reinit parameters of final layer
                # 3) Copy saved parameters to new final layer
                known_class_ids = torch.LongTensor(list(known_classes))
                saved_params = [p.data.clone() for p in model.final_parameters()]
                model.reset_final_parameters()
                print(known_class_ids)
                for i, params in enumerate(model.final_parameters()):
                    if params.dim() == 1:  # bias vector
                        params.data[known_class_ids] = saved_params[i][known_class_ids]
                    elif params.dim() == 2:  # weight matrix
                        params.data[known_class_ids, :] = saved_params[i][known_class_ids, :]
                    else:
                        NotImplementedError("Parameter dim > 2 ?")
                del saved_params  # Explicit cleanup!?
        else:
            raise NotImplementedError("Unknown --start arg: '%s'" % args.start)

        known_classes |= new_classes
        print(f"Known classes at time {current_year}:", known_classes)

        if has_parameters:
            # Build a fresh optimizer in both cases: warm or cold
            # Use rescaled lr and wd
            if args.model == 'node2vec':
                # Use SparseAdam for node2vec to speed things up
                optimizer = torch.optim.SparseAdam(model.parameters(),
                                                   lr=args.lr)
            else:
                optimizer = torch.optim.Adam(model.parameters(),
                                             lr=args.lr * args.rescale_lr,
                                             weight_decay=args.weight_decay * args.rescale_wd)
        if args.model == 'mostfrequent':
            if epochs > 0:
                # Re-fit only if uptraining is in general allowed!
                model.fit(None, subg_labels[train_nid])

            acc, f1, _ = evaluate(model,
                              subg,
                              subg_features,
                              subg_labels,
                              mask=test_nid,
                              compute_loss=False)
        elif args.model == 'node2vec':
            train_node2vec(model, optimizer, epochs=epochs,
                           batch_size=args.n2v_batch_size,
                           shuffle=True,
                           num_workers=args.n2v_num_workers)
            acc = evaluate_node2vec(model, subg_labels, train_nid, test_nid)

        elif args.model == "graphsaint":
            if epochs > 0:
                print("Training SAINT inductively")
                train_saint(model,
                            optimizer,
                            subg,
                            subg_features,
                            subg_labels,
                            sampling=args.sampling,
                            mask=None,
                            epochs=epochs,
                            weights=weights,
                            walk_length=args.walk_length,
                            batch_size=args.batch_size,
                            coverage=args.saint_coverage,
                            n_jobs=saint_njobs,
                            device=device)
            subg, subg_features, subg_labels, subg_years, train_nid, test_nid = prepare_data_for_year(graph,
                                                                                                      features,
                                                                                                      labels,
                                                                                                      years,
                                                                                                      current_year,
                                                                                                      args.history,
                                                                                                      device=device,
                                                                                                      backend=backend)

            acc, f1, _ = evaluate_saint(model,
                                    subg,
                                    subg_features,
                                    subg_labels,
                                    mask=test_nid,
                                    compute_loss=False)
        else:
            if epochs > 0:
                train(model,
                      optimizer,
                      subg,
                      subg_features,
                      subg_labels,
                      mask=train_nid,
                      epochs=epochs,
                      weights=weights,
                      backend=backend)

            acc, f1, _ = evaluate(model,
                              subg,
                              subg_features,
                              subg_labels,
                              mask=test_nid,
                              compute_loss=False,
                              backend=backend)
        print(f"[{current_year} ~ Epoch {epochs}] Test Accuracy: {acc:.4f}")
        results_df = attach_score(results_df, current_year.item(), epochs, acc, f1)
        # input() # debug purposes
        # DROP ALL STUFF COMPUTED FOR CURRENT WINDOW (no memory leaks)
        del subg
        del subg_features
        del subg_labels
        del subg_years
        del train_nid
        del test_nid
        del data
        gc.collect()

        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             print(type(obj), obj.size())
        #     except:
        #         pass
        # input()
        # torch.cuda.empty_cache()

    if args.save is not None:
        print("Saving final results to", args.save)
        appendDFToCSV_void(results_df, args.save)


DATASET_PATHS = {
    'dblp-easy': os.path.join('data', 'dblp-easy'),
    'dblp-hard': os.path.join('data', 'dblp-hard'),
    'pharmabio': os.path.join('data', 'pharmabio'),
    'dblp-full': os.path.join('data', 'dblp-full')
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="Specify model", default='gs-mean',
                        choices=['mlp', 'gs-mean', 'mostfrequent',
                                 'egcn', 'gat', 'jknet', 'graphsaint',
                                 'node2vec', 'sgnet'])
    parser.add_argument('--sampling', type=str, choices=['rw', 'node', 'edge'],
                        default=None)
    parser.add_argument('--variant', type=str, default='',
                        help="Model variant, if model is GraphSAINT, specifies the Geometric base model")
    parser.add_argument('--dataset', type=str, help="Specify the dataset", choices=list(DATASET_PATHS.keys()),
                        default='pharmabio')
    parser.add_argument('--t_start', type=int,
                        help="The first evaluation time step. Default is 2004 for DBLP-{easy,hard} and 1999 for PharmaBio")

    parser.add_argument('--n_layers', type=int,
                        help="Number of layers/hops", default=2)
    parser.add_argument('--n_hidden', type=int,
                        help="Model dimension", default=64)
    parser.add_argument('--lr', type=float,
                        help="Learning rate", default=0.01)
    parser.add_argument('--weight_decay', type=float,
                        help="Weight decay", default=0.0)
    parser.add_argument('--dropout', type=float,
                        help="Dropout probability", default=0.5)

    parser.add_argument('--initial_epochs', type=int,
                        help="Train this many initial epochs", default=0)
    parser.add_argument('--annual_epochs', type=int,
                        help="Train this many epochs per year", default=200)
    parser.add_argument('--history', type=int,
                        help="How many years of data to keep in history", default=100)

    parser.add_argument('--gat_out_heads',
                        help="How many output heads to use for GATs", default=1, type=int)
    parser.add_argument('--rescale_lr', type=float,
                        help="Rescale factor for learning rate and weight decay after pretraining", default=1.)
    parser.add_argument('--rescale_wd', type=float,
                        help="Rescale factor for learning rate and weight decay after pretraining", default=1.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_neighbors', type=int, default=1,
                        help="How many neighbors for control variate sampling")
    parser.add_argument('--limit', type=int, default=None,
                        help="Debug mode, limit number of papers to load")
    parser.add_argument('--batch_size', type=str, default="1000",
                        help="Number of seed nodes per batch for sampling")
    parser.add_argument('--test_batch_size', type=int, default=10000,
                        help="Test batch size (testing is done on cpu)")
    parser.add_argument('--num_workers', type=int, default=8, help="How many threads to use for sampling")
    parser.add_argument('--limited_pretraining', default=False, action="store_true",
                        help="Perform pretraining on the first history window.")
    parser.add_argument('--decay', default=None, type=float, help="Paramater for exponential decay loss smoothing")
    parser.add_argument('--save_intermediate', default=False, action="store_true",
                        help="Save intermediate results per year")
    parser.add_argument('--save', default=None, help="Save results to this file")
    parser.add_argument('--start', default='legacy-warm',
                        choices=['cold', 'warm', 'hybrid', 'legacy-cold', 'legacy-warm'],
                        help="Cold retrain from scratch or use warm start.")
    parser.add_argument("--walk_length", default=2, type=int, help="Walk length for GraphSAINT random walk sampler")
    parser.add_argument("--saint_coverage", default=500, type=int, help="Compute normalization statistics with this much coverage")
    parser.add_argument("--saint_njobs", type=int, default=1, help="Number of jobs to sample for GraphSAINT")
    parser.add_argument("--backend", choices=["dgl", "geometric"], help="Backend to use", default='dgl')

    add_node2vec_args(parser)

    ARGS = parser.parse_args()


    if ARGS.batch_size.isdigit():
        ARGS.batch_size = int(ARGS.batch_size)
        print("Using an absolute batch size of", ARGS.batch_size, "for GraphSAINT")
    else:
        ARGS.batch_size = float(ARGS.batch_size)
        print("Using a relative batch size of", ARGS.batch_size, "for GraphSAINT")


    if ARGS.save is None:
        print("**************************************************")
        print("*** Warning: results will not be saved         ***")
        print("*** consider providing '--save <RESULTS_FILE>' ***")
        print("**************************************************")

    # Handle dataset argument to get path to data
    try:
        ARGS.data_path = DATASET_PATHS[ARGS.dataset]
    except KeyError:
        print("Dataset key not found, trying to interprete as raw path")
        ARGS.data_path = ARGS.dataset
    print("Using dataset with path:", ARGS.data_path)

    # Handle t_start argument
    if ARGS.t_start is None:
        try:
            ARGS.t_start = {
                'dblp-easy': 2004,
                'dblp-hard': 2004,
                'pharmabio': 1999,
                'dblp-full': 2004
            }[ARGS.dataset]
            print("Using t_start =", ARGS.t_start)
        except KeyError:
            print("No default for dataset '{}'. Please provide '--t_start'.".format(ARGS.dataset))
            exit(1)

    # Backward compatibility:
    # current implementation actually uses 'pretrain_until'
    # as last timestep / year *BEFORE* t_start
    ARGS.pretrain_until = ARGS.t_start - 1


    # Sanity checks #
    if ARGS.model == 'node2vec':
        # Sanity checks
        if 'warm' in ARGS.start:
            raise NotImplementedError("Node2vec with warm starts is not yet supported")
        else:
            ARGS.start = 'legacy-cold'
            print(f"Using '{ARGS.start}' restart mode for Node2Vec.")

    main(ARGS)
