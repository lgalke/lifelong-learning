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
from models.node2vec import (add_node2vec_args,
                             train_node2vec,
                             evaluate_node2vec)

# from datasets import load_data  # unused

from lifelong_learning import lifelong_nodeclf_identifier
from lifelong_learning import LifelongNodeClassificationDataset
from lifelong_learning import collate_tasks

from resultswriter import CSVResultsWriter

import open_learning

try:
    import wandb
    USE_WANDB = True
except ImportError:
    USE_WANDB = False
    print("Not using weightsandbiases integration. To use `pip install wandb`")




def compute_weights(ts, exponential_decay, initial_quantity=1.0, normalize=True):
    ts = torch.as_tensor(ts)
    delta_t = ts.max() - ts
    values = initial_quantity * torch.exp(- exponential_decay * delta_t)
    if normalize:
        # When normalizing, the initial_quantity is irrelevant
        values = values / values.sum()
    return values


def train(model, optimizer, g, feats, labels, mask=None, epochs=1,
          weights=None, backend='dgl', open_learning_model=None):
    model.train()
    reduction = 'none' if weights is not None else 'mean'

    if hasattr(model, '__reset_cache__'):
        print("Resetting Model Cache")
        model.__reset_cache__()

    if mask is not None:
        # Reduce view alreayd here rather than in each epoch (prevent bugs)
        labels = labels[mask]

    for epoch in range(epochs):
        inputs = (g, feats) if backend == 'dgl' else (feats, g)

        logits = model(*inputs)
        if mask is not None:
            logits = logits[mask]

        if open_learning_model is not None:
            # The open learning model defines the loss
            # print("Logits", logits.size(), logits.dtype)
            # print("Labels", labels.size(), labels.dtype)
            loss = open_learning_model.loss(logits, labels)
        else:
            # Standard cross entropy training
            loss = F.cross_entropy(logits, labels, reduction=reduction)

        if weights is not None:
            loss = (loss * weights).sum()

        # Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        myloss = loss.detach().item()
        myepoch = epoch + 1
        wandb.log({"epoch": myepoch, "train/loss": myloss})
        print("\rEpoch {:d} | Loss: {:.4f}".format(myepoch, myloss),
              flush=True, end='')

    if open_learning_model is not None:
        print("Fitting Open Learning Model")
        open_learning_model.fit(logits, labels)
        print(open_learning_model)


def evaluate(model, g, feats, labels, mask=None, compute_loss=True,
             backend='dgl',
             open_learning_model=None, unseen_classes: set = None):
    model.eval()

    if hasattr(model, '__reset_cache__'):
        print("Resetting Model Cache")
        model.__reset_cache__()

    with torch.no_grad():
        inputs = (g, feats) if backend == 'dgl' else (feats, g)
        logits = model(*inputs)

        # Reduce view on test mask
        if mask is not None:
            logits = logits[mask]
            labels = labels[mask]

        if compute_loss:
            if open_learning_model is None:
                loss = F.cross_entropy(logits, labels).item()
            else:
                loss = open_learning_model.loss(logits, labels).item()

        if isinstance(logits, np.ndarray):
            logits = torch.FloatTensor(logits)
        __max_vals, max_indices = torch.max(logits.detach(), 1)
        acc = (max_indices == labels).sum().float() / labels.size(0)
        f1 = f1_score(labels.cpu(), max_indices.cpu(), average="macro")

        scores = {
            'accuracy': acc.item(),
            'f1_macro': f1,
            'loss': loss
        }

        if open_learning_model is not None:
            reject_mask = open_learning_model.reject(logits)
            predictions = open_learning_model.predict(logits)
            open_scores = open_learning.evaluate(labels, unseen_classes,
                                                 predictions, reject_mask)
            scores.update(open_scores)

    # return acc.item(), f1, loss
    return scores


def build_model(args, in_feats, n_hidden, n_classes, device, n_layers=1, backend='geometric'):
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
            raise NotImplementedError("Node2vec initializer needs to move to different location")
            # model = tg.nn.Node2Vec(
            #     edge_index,
            #     n_hidden,
            #     args.n2v_walk_length,
            #     args.n2v_context_size,
            #     walks_per_node=args.n2v_walks_per_node,
            #     p=args.n2v_p,
            #     q=args.n2v_q,
            #     num_negative_samples=args.n2v_num_negative_samples,
            #     num_nodes=num_nodes,
            #     sparse=True
            # )
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


def build_optimizer(args, model):
    if args.model in ['most_frequent']:
        # for models that don't need an optimizer
        return None

    if args.model == 'node2vec':
        # Use SparseAdam for node2vec to speed things up
        optimizer = torch.optim.SparseAdam(model.parameters(),
                                           lr=args.lr * args.rescale_lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr * args.rescale_lr,
                                     weight_decay=args.weight_decay * args.rescale_wd)
    return optimizer

def count_params(model):
   return sum(np.product(p.size()) for p in model.parameters())

def restart(model, mode, known_classes: set, new_classes: set):
    if mode == 'cold' or (mode == 'hybrid' and new_classes):
        # NEW version, equivalent to legacy-cold, but more efficient
        model.reset_parameters()
    elif mode == 'warm':
        # Skip for first task (does not make sense and makes problem for SGNET)
        if new_classes:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("~~~~~~ New classes encountered... ~~~~~~")
            print("~~~~~~ doing partial warm reinit! ~~~~~~")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            # If there are new classes:
            # 1) Save parameters of final layer
            # 2) Reinit parameters of final layer
            # 3) Copy saved parameters to new final layer
            known_class_ids = torch.LongTensor(list(known_classes))
            saved_params = [p.data.clone() for p in model.final_parameters()]
            model.reset_final_parameters()
            print("[Debug] known_class_ids during restart:", known_class_ids)
            for i, params in enumerate(model.final_parameters()):
                if params.dim() == 1:  # bias vector
                    params.data[known_class_ids] = saved_params[i][known_class_ids]
                elif params.dim() == 2:  # weight matrix
                    params.data[known_class_ids, :] = saved_params[i][known_class_ids, :]
                else:
                    NotImplementedError("Parameter dim > 2 ?")
            # del saved_params  # Explicit cleanup!?
    else:
        raise NotImplementedError("Unknown --start arg: '%s'" % mode)
    return model

def zero_unseen_classes(model, unseen_classes: set):
    print(f"Setting params to zero for {len(unseen_classes)} classes")
    unseen_class_ids = torch.LongTensor(list(unseen_classes))
    for params in model.final_parameters():
        if params.dim() == 1:  # bias vector
            params.data[unseen_class_ids] = -1e12  # big negative bias
        elif params.dim() == 2:  # weight matrix
            params.data[unseen_class_ids, :] = 0   # zero weights
        else:
            NotImplementedError("Parameter dim > 2 ?")

    return model





def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_sampling = args.model in ['gcn_cv_sc']
    backend = args.backend

    print("Using backend:", backend)

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if args.model == 'mostfrequent':
        device = torch.device("cpu")

    # LEGACY CODE, not used anymore
    # if args.model in ['graphsaint']:
    #     print("///////////////////")
    #     print("//// inductive ////")
    #     print("///////////////////")
    #     # Train completely on Task t-1
    #     globals_device = torch.device("cpu")
    #     assert args.inductive
    # else:
    #     print("//////////////////////")
    #     print("//// transductive ////")
    #     print("//////////////////////")
    #     globals_device = device
    #     inductive = False

    # Assume preprocessed dataset is in subdir of dataset
    print("Expecting preprocessed data at", args.data_path)
    dataset = LifelongNodeClassificationDataset(args.data_path,
                                                inductive=args.inductive)
    print(dataset)
    print(f"[t_min, tmax] = [{dataset.t_min}, {dataset.t_max}]")
    print(f"t_zero in dataset = {dataset.t_zero} (should be the one before t_start)")
    assert dataset.t_zero == args.t_start - 1, "Supplied t_start -1 is not equal to t_zero of dataset"
    assert dataset.history_size == args.history, "History sizes do not match"
    assert dataset.backend == args.backend, "Backends do not match"

    n_classes = dataset.num_classes
    in_feats = dataset.num_features
    n_hidden = args.n_hidden

    model = build_model(args, in_feats, n_hidden, n_classes, device,
                        n_layers=args.n_layers, backend=backend)
    if args.model == 'gcn_cv_sc':
        # unzip training and inference models
        model, infer_model = model
    print(model)
    optimizer = build_optimizer(args, model)

    if USE_WANDB:
        wandb.watch(model)

    num_params = count_params(model) if optimizer is not None else 0
    print("#params:", num_params)
    if args.only_count_params:
        exit(0)


    rw = CSVResultsWriter(args)

    known_classes = set()
    all_classes = set(range(dataset.num_classes))
    taskloader = torch.utils.data.DataLoader(dataset, shuffle=False,
                                             batch_size=1,
                                             collate_fn=collate_tasks)

    if args.open_learning is not None:
        olg_model = open_learning.build(args)
        print("Open Learning Model:", olg_model)
    else:
        # backward compat
        olg_model = None

    for t, batch in enumerate(taskloader):
        if args.only_first_task and t > 0:
            print("Finished with first task, exiting.")
            break

        if args.inductive:
            train_task, task = batch[0]
        else:
            train_task = None
            task = batch[0]
        current_year = task.task_id

        print("Batch:", batch)
        print("Task:", task)
        print("Train mask:", task.train_mask.size())
        print("Test mask:", task.test_mask.size())
        print("Feats:", task.x.size())
        print("Labels:", task.y.size())

        if args.decay is not None:
            if args.inductive:
                raise NotImplementedError("Decay only implemented for transductive learning")
            # Use decay factor to weight the loss function based on time steps t
            if use_sampling:
                raise NotImplementedError("Decay can only be used without sampling")
            weights = compute_weights(task.task_ids[task.train_mask], args.decay, normalize=True).to(device)
        else:
            weights = None

        # Do the pretraining on the first history window
        # with `initial_epochs` instead of `annual_epochs`
        epochs = args.initial_epochs if t == 0 else args.annual_epochs
        print(f"Training {epochs} epochs for task {t+1} (year {current_year})")

        print(f"Known classes at time {current_year}:", known_classes)
        # Find new classes
        if args.inductive:
            # Task is used completely for training
            new_classes = set(train_task.y.numpy()) - known_classes
            # unseen_classes = set(task.y.numpy()) - known_classes - new_classes
        else:
            new_classes = set(task.y[task.train_mask].numpy()) - known_classes
            # unseen_classes = set(task.y[task.test_mask].numpy()) - known_classes - new_classes

        print(f"New classes at train time {current_year}:", new_classes)

        # Perform a restart (beginning with 2nd task)
        if t > 0:
            restart(model, args.start, known_classes, new_classes)
        # Add new classes to known classes
        known_classes |= new_classes

        # All classes that are not in the training set of t are unseen
        unseen_classes = all_classes - known_classes
        print(f"Unseen classes at test time {current_year}:", unseen_classes)

        test_loss = None  # fall-back if evaluate model doesn't emit loss

        if args.model == 'mostfrequent':
            assert args.subsample_train is None, "MostFrequent not impl. for subsample train"
            assert args.open_learning is None, "Open Learning not impl. for mostfrequent"
            assert args.inductive
            if epochs > 0:
                # Re-fit only if uptraining is in general allowed!
                model.fit(None, train_task.y)
            del train_task
            scores = evaluate(model,
                              task.graph(),
                              task.x,
                              task.y,
                              mask=task.test_mask,
                              compute_loss=False)
            acc, f1 = scores['accuracy'], scores['f1_macro']
        elif args.model == 'node2vec':
            assert args.subsample_train is None, "MostFrequent not impl. for subsample train"
            assert not args.inductive, "Node2vec can only be applied transductively"
            assert args.open_learning is None, "Open Learning not impl. for node2vec"
            train_node2vec(model, optimizer, epochs=epochs,
                           batch_size=args.n2v_batch_size,
                           shuffle=True,
                           num_workers=args.n2v_num_workers)
            acc = evaluate_node2vec(model, task.y, task.train_mask, task.test_mask)

        elif args.model == "graphsaint":
            # DON'T shift to GPU for graphsaint, it WILL fail
            assert args.inductive, "GraphSAINT is only implemented for the inductive case"
            assert args.subsample_train is None, "Subsample Train (label rate) not impl. for GraphSAINT"
            assert args.open_learning is None, "Open Learning not impl. for GraphSAINT"
            train_saint(model,
                        optimizer,
                        train_task.graph(),
                        train_task.x,
                        train_task.y,
                        sampling=args.sampling,
                        mask=None,
                        epochs=epochs,
                        weights=weights,
                        walk_length=args.walk_length,
                        batch_size=args.batch_size,
                        coverage=args.saint_coverage,
                        n_jobs=1,
                        device=device)
            del train_task
            torch.cuda.empty_cache()
            if args.evaluate_saint_on_cpu:
                # Shift data toGPU
                model = model.cpu()
                print("Evaluating SAINT on CPU")
            else:
                task = task.to(device)
                # Shift model to CPU
            acc, f1, test_loss = evaluate_saint(model,
                                    task.graph(),
                                    task.x,
                                    task.y,
                                    mask=task.test_mask,
                                    compute_loss=True)
            if args.evaluate_saint_on_cpu:
                # Shift model back to gpu
                model = model.to(device)
            gc.collect()
            torch.cuda.empty_cache()
        else:
            if args.inductive:
                assert args.subsample_train is None, "Inductive not impl. for subsample train"
                # Train on t-1
                train_task = train_task.to(device)
                train(model,
                      optimizer,
                      train_task.graph(),
                      train_task.x,
                      train_task.y,
                      mask=None,
                      epochs=epochs,
                      weights=weights,
                      backend=backend,
                      open_learning_model=olg_model)
                del train_task
                gc.collect()
                torch.cuda.empty_cache()

            # Put current task on device
            task = task.to(device)

            if not args.inductive:
                # Train on train_mask of current task
                train(model,
                      optimizer,
                      task.graph(),
                      task.x,
                      task.y,
                      mask=task.train_mask,
                      epochs=epochs,
                      weights=weights,
                      backend=backend,
                      open_learning_model=olg_model)

            # acc, f1, test_loss = evaluate(model,  # <- old

            # inplace
            # model = zero_unseen_classes(model, unseen_classes)

            scores = evaluate(model,
                              task.graph(),
                              task.x,
                              task.y,
                              mask=task.test_mask,
                              compute_loss=True,
                              backend=backend,
                              open_learning_model=olg_model,
                              unseen_classes=unseen_classes)

        # print(f"[{current_year} ~ Epoch {epochs}] Test Accuracy: {acc:.4f}")
        print(f"[{current_year} ~ Epoch {epochs}] Scores: {scores}")

        assert 'year' not in scores
        assert 'epoch' not in scores
        scores['year'] = current_year
        scores['epoch'] = epochs

        # results_df = attach_score(results_df, current_year, epochs, scores)

        rw.add_result(scores)

        if USE_WANDB:
            # Prefix with 'test/' to improve structure in wandb dashboard
            log_dict = {'test/'+k: v for k, v in scores.items()}
            log_dict["task_id"] = current_year
            log_dict["task_index"] = t
            wandb.log(log_dict)

        # input() # debug purposes
        # DROP ALL STUFF COMPUTED FOR CURRENT WINDOW (no memory leaks)
        del task
        gc.collect()
        torch.cuda.empty_cache()

        # Memory leak debugging, not needed.
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             print(type(obj), obj.size())
        #     except:
        #         pass
        # input()

    if USE_WANDB:
        # This makes WandB compute summary metrics for accuracy and f1 macro
        # including the average!
        # TODO: be careful when more than one accuracy per task is stored in results
        # (currently not a problem, as we only store one set of scores per task)
        wandb.run.summary["test/avg_accuracy"] = rw.data["accuracy"].values.mean()
        wandb.run.summary["test/sd_accuracy"] = rw.data["accuracy"].values.std(ddof=1)
        wandb.run.summary["test/avg_f1_macro"] = rw.data["f1_macro"].values.mean()
        wandb.run.summary["test/sd_f1_macro"] = rw.data["f1_macro"].values.std(ddof=1)

        wandb.run.summary["test/avg_open_f1_macro"] = rw.data["open_f1_macro"].values.mean()
        wandb.run.summary["test/sd_open_f1_macro"] = rw.data["open_f1_macro"].values.std(ddof=1)
        wandb.run.summary["test/avg_open_mcc"] = rw.data["open_mcc"].values.mean()
        wandb.run.summary["test/sd_open_mcc"] = rw.data["open_mcc"].values.std(ddof=1)
        # wandb.run.summary.update()

        wandb.run.summary["test/open_tp"] = rw.data["open_tp"].values.sum()
        wandb.run.summary["test/open_tn"] = rw.data["open_tn"].values.sum()
        wandb.run.summary["test/open_fp"] = rw.data["open_fp"].values.sum()
        wandb.run.summary["test/open_fn"] = rw.data["open_fn"].values.sum()

    if args.save is not None:
        print("Saving final results to", args.save)
        # appendDFToCSV_void(results_df, args.save)

        rw.write(args.save)


DATASET_PATHS = {
    'dblp-easy': os.path.join('data', 'dblp-easy'),
    'dblp-hard': os.path.join('data', 'dblp-hard'),
    'pharmabio': os.path.join('data', 'pharmabio'),
    'dblp-full': os.path.join('data', 'dblp-full')
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="Specify model", default='gs-mean',
                        choices=['mlp', 'gs-mean', 'gcn_cv_sc', 'mostfrequent',
                                 'egcn', 'gat', 'gcn', 'jknet-sageconv', 'jknet-graphconv', 'graphsaint',
                                 'node2vec', 'sgnet'])
    parser.add_argument('--sampling', type=str, choices=['rw', 'node', 'edge'],
                        default=None, help="Sampling strategy. Only for GraphSAINT")
    parser.add_argument('--variant', type=str, default='',
                        help="Model variant, if model is GraphSAINT, specifies the Geometric base model")
    parser.add_argument('--dataset', type=str, help="Specify the dataset", # choices=list(DATASET_PATHS.keys()),
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

    parser.add_argument('--initial_epochs', type=int, help="Train this many epochs on first task (defaults to annual epochs)", default=None)
    parser.add_argument('--annual_epochs', type=int, help="Train this many epochs on all subsequent tasks", default=200)
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
    # parser.add_argument('--limited_pretraining', default=False, action="store_true",
    #                     help="Perform pretraining on the first history window.")
    parser.add_argument('--decay', default=None, type=float, help="Paramater for exponential decay loss smoothing")
    parser.add_argument('--save_intermediate', default=False, action="store_true",
                        help="Save intermediate results per year")
    parser.add_argument('--save', default=None, help="Save results to this file")
    parser.add_argument('--start', default='legacy-warm',
                        choices=['cold', 'warm', 'hybrid', 'legacy-cold', 'legacy-warm'],
                        help="Cold retrain from scratch or use warm start.")
    parser.add_argument("--walk_length", default=2, type=int, help="Walk length for GraphSAINT random walk sampler")
    parser.add_argument("--saint_coverage", default=0, type=int)
    parser.add_argument("--backend", choices=["dgl", "geometric"], help="Backend to use", default='dgl')
    parser.add_argument("--inductive", default=False, action='store_true', help="Train on task t-1, then eval on test set of task t")
    parser.add_argument("--only_first_task", default=False, action='store_true', help="Train only on first task (debug purposes)")
    parser.add_argument("--only_count_params", default=False, action='store_true', help="Print number of parameters and exit (debug purposes)")
    parser.add_argument("--evaluate_saint_on_cpu", default=False, action='store_true', help="Run the eval step of GraphSAINT on CPU")
    parser.add_argument('--comment', type=str, default='', help="Some comment for logging purposes.")
    parser.add_argument('--label_rate', type=float, default=None, help="Label rate (needs to be preprocessed)")
    add_node2vec_args(parser)

    open_learning.add_args(parser)

    ARGS = parser.parse_args()

    if USE_WANDB:
        wandb.init(project="lifelong-learning")
        wandb.config.update(ARGS)


    if ARGS.initial_epochs is None:
        ARGS.initial_epochs = ARGS.annual_epochs


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
        dataset_path = DATASET_PATHS[ARGS.dataset]

        preprocessed_dataset_identifier = lifelong_nodeclf_identifier(ARGS.dataset, ARGS.t_start-1, ARGS.history, ARGS.backend, label_rate=ARGS.label_rate)
        ARGS.data_path = os.path.join(dataset_path, preprocessed_dataset_identifier)
    except:
        print(f"Dataset not in dict, assuming preprocessed dataset at: {ARGS.dataset}")
        ARGS.data_path = ARGS.dataset
    print("Using dataset with path:", ARGS.data_path)

    # Handle t_start argument
    if ARGS.t_start is None:
        try:
            ARGS.t_start = {
                'dblp-easy': 2004,
                'dblp-hard': 2004,
                'pharmabio': 1999
            }[ARGS.dataset]
            print("Using t_start =", ARGS.t_start)
        except KeyError:
            print("No default for dataset '{}'. Please provide '--t_start'."
                  .format(ARGS.dataset))
            exit(1)

    # Backward compatibility:
    # current implementation actually uses 'pretrain_until'
    # as last timestep / year *BEFORE* t_start
    # ARGS.pretrain_until = ARGS.t_start - 1
    # Not needed anymore

    # Sanity checks #
    if ARGS.model == 'node2vec':
        # Sanity checks
        if 'warm' in ARGS.start:
            raise NotImplementedError("Node2vec w/ warm starts not supported")
        else:
            ARGS.start = 'legacy-cold'
            print(f"Using '{ARGS.start}' restart mode for Node2Vec.")
    elif ARGS.model == 'graphsaint':
        assert ARGS.inductive, "GraphSAINT only works for inductive mode"

    main(ARGS)
