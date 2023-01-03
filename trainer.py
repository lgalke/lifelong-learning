import os.path

import dataclasses
from dataclasses import dataclass, field, is_dataclass

from tqdm import tqdm
import numpy as np
import pandas as pd
from enum import Enum
from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F

from lifelong_learning import LifelongGraphDataset

class RestartMode(Enum):
    WARM = "warm"
    COLD = "cold"


@dataclass
class TrainingArguments:
    dataset: str = field(default=None, metadata={"help": "Dataset name or path"})
    history: int = field(default=None, metadata={"help": "History size"})
    restart_mode: RestartMode = field(default="warm", metadata={"help": "Restart mode of {'warm', 'cold'}"})
    learning_rate: float = field(default=1e-3, metadata={"help": "The initial learning rate for Adam."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay if we apply some."})
    dropout: float = field(default=0.1, metadata={"help": "Dropout probability"})
    num_pretrain_epochs: int = field(default=0, metadata={"help": "Number of pretraining epochs"})
    num_train_epochs: int = field(default=200, metadata={"help": "Number of training epochs per time"})


class ResultsWriter:
    """
    A stateful ResultsWriter that writes results to csv files including hyperparameters and other things
        Example:
            rw = ResultsWriter("something",
    """
    def __init__(self, path: str, state: dict = None):
        self.path = path
        self._state = dict(state) if state is not None else {}
        self._frozen_keys = None

    def _freeze(self, keys):
        """ Freezes the keys used for the first write """
        self._frozen_keys = frozenset(keys)

    @property
    def _isfrozen(self):
        return bool(self._frozen_keys)

    def _check(self, keys):
        if self._isfrozen and not set(keys).issubset(self._frozen_keys):
            raise KeyError("ResultsWriter's keys are frozen")

    def update(self, dictlike:dict=None, **kwargs):
        """
        Safely updates internal state, for example: rw.update(t=2)
        """
        if dictlike:
            kwargs = {**dictlike, **kwargs}
        self._check(kwargs.keys())

        self._state.update(kwargs)

    def add_result(self, dictlike:dict=None, **kwargs):
        """
        Writes a result to `self.path`. Arguments can be provided as dict or as keyword arguments.
        """
        if dictlike:
            kwargs = {**dictlike, **kwargs}
        self._check(kwargs.keys())

        # Create new dict to write, *dont* update state
        record = {**self._state, **kwargs}

        # Write full record to csv file
        result = pd.DataFrame.from_records([record])
        # include header only if file does not exist
        header = not os.path.isfile(self.path)
        result.to_csv(self.path, header=header, index=False, mode='a')

        if not self._isfrozen:
            # Freeze after first write
            # including result-specific columns
            self._freeze(record.keys())
            # No more updates to the state's keys are allowed


class IncrementalTrainer:
    def __init__(self, model, dataset: LifelongGraphDataset, args: TrainingArguments):
        """ Initializes a trainer """
        # TODO: Add argument for inductive
        # TODO: Add tensorboard SummaryWriter
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.args = args
        self.dataset = dataset
        self.num_nodes = features.size(0) if num_nodes is None else num_nodes
        assert args.restart_mode in ['warm', 'cold'], "Unknown restart mode: " + restart_mode
        self.results_writer = ResultsWriter("/tmp/test.csv", dataclasses.asdict(args))
        self.optimizer = self._build_optimizer()

    def _build_optimizer(self):
        """
        Method that returns an optimizer
        Subclasses may overwrite this to use a different optimizer
        """
        return torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay)

    def _prepare_data_for_time(self, t, history, device=None, exclude_class=None):
        print("Preparing data for time:", t)
        # Prepare subgraph

        # Subg holds vertex ids corresponding to original graph
        subg_nodes = torch.arange(self.num_nodes)[(self.timestamps <= t) & (t >= (t - history))]
        subg_num_nodes = subg_nodes.size(0)

        # Create the subgraph (depends on backend)
        if self.backend == 'dgl':
            subg = graph.subgraph(subg_nodes)
            subg.set_n_initializer(dgl.init.zero_initializer)
        elif self.backend == 'geometric':
            subg, __edge_attr = tg.utils.subgraph(subg_nodes,
                                                  graph, relabel_nodes=True)
        else:
            raise ValueError("Unkown backend: " + backend)

        # Filter supplementary data for subgraph vertices
        subg_features = self.features[subg_nodes]
        subg_labels = self.labels[subg_nodes]
        subg_timestamps = self.timestamps[subg_nodes]

        # Prepare masks wrt *subgraph*
        train_nid = torch.arange(subg_num_nodes)[subg_timestamps < t]
        test_nid = torch.arange(subg_num_nodes)[subg_timestamps == t]

        if exclude_class is not None:
            train_nid = train_nid[subg_labels[train_nid] != exclude_class]
            test_nid = test_nid[subg_labels[test_nid] != exclude_class]

        print("[{}] #Training: {}".format(t, train_nid.size(0)))
        print("[{}] #Test    : {}".format(t, test_nid.size(0)))
        if device is not None:
            if self.backend == 'geometric':
                subg = subg.to(device)
            subg_features = subg_features.to(device)
            subg_labels = subg_labels.to(device)
        return subg, subg_features, subg_labels, subg_timestamps, train_nid, test_nid

    def _prepare_data_for_time_inductive(self, t, history, **kwargs):
        train_g, train_feats, train_labels, train_timestamps, __, __ = self._prepare_data_for_time(t-1, history, **kwargs)
        test_g, test_feats, test_labels, test_timestamps, __, test_mask = self._prepare_data_for_time(t, history, **kwargs)
        return train_data, test_data

    def restart(self, known_classes:set=None, new_classes:set=None):
        """ Performs a restart in-between different time steps """
        # TODO integrate hybrid strategy
        if self.args.restart_mode == 'cold':
            # cold restart -> reset all parameters
            self.model.reset_parameters()
        elif self.args.restart_mode == 'warm':
            if known_classes and new_classes:
                # If we have both known and new classes
                # Copy all parameters for known classes
                # and freshly initialize params for new classes
                # **Models must implement `final_parameters()` and `reset_final_parameters()`**
                known_class_ids = torch.LongTensor(list(known_classes))
                saved_params = [p.data.clone() for p in self.model.final_parameters()]
                self.model.reset_final_parameters()
                for i, params in enumerate(self.model.final_parameters()):
                    if params.dim() == 1:  # bias vector
                        params.data[known_class_ids] = saved_params[i][known_class_ids]
                    elif params.dim() == 2:  # weight matrix
                        params.data[known_class_ids, :] = saved_params[i][known_class_ids, :]
                    else:
                        NotImplementedError("Parameter dim > 2 ?")
            # Else do nothing, but keep model parameters as they are
        else:
            raise NotImplementedError("Unknown restart mode '%s':" % self.restarts)

        # Reset the state of the optimizer
        self.optimizer = self._build_optimizer()

    def _training_step(self, g, feats, labels, mask=None, weights=None):
        """ Perform one training step """
        inputs = (g, feats) if self.backend == 'dgl' else (feats, g)
        logits = self.model(*inputs)
        reduction = 'none' if weights is not None else 'mean'

        if mask is not None:
            loss = F.cross_entropy(logits[mask], labels[mask], reduction=reduction)
        else:
            loss = F.cross_entropy(logits, labels, reduction=reduction)

        if weights is not None:
            loss = (loss * weights).sum()

        # Step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def _train_epoch(self, *args, **kwargs):
        # More flexible subclassing
        return self._training_step(*args, **kwargs)

    def train(self, g, feats, labels, mask=None, weights=None):
        """ Train multiple epochs """
        self.model.train()
        for epoch in tqdm(range(self.args.num_train_epochs), desc="Epoch"):
            loss = self._train_epoch(g, feats, labels, mask=mask)
            print("Epoch {:d} | Loss: {:.4f}".format(epoch + 1, loss.detach().item()))

    def evaluate(self, g, feats, labels, mask=None, compute_loss=True):
        # TODO: Store results
        self.model.eval()
        with torch.no_grad():
            inputs = (g, feats) if self.backend == 'dgl' else (feats, g)
            logits = self.model(*inputs)

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

    def train_and_evaluate_incremental(self, t_start, history):
        """ Trains and evaluates incrementally starting at `t_start` (inclusive)"""
        t_end = self.timestamps.max()
        ts = torch.unique(self.timestamps[self.timestamps >= t_start], sorted=True)
        known_classes = set()
        for t in tqdm(ts, desc="Time"):
            # subggraph
            g, x, y, __, train_nid, test_nid = self._prepare_data_for_time(t, history)
            new_classes = set(y[train_nid].cpu().numpy()) - known_classes
            self.restart(known_classes, new_classes)
            self.train(g, x, y, mask=train_nid, epochs=self.epochs_per_time)
            acc, f1, loss = self.evaluate(g, x, y, mask=test_nid, compute_loss=True)
            known_classes |= new_classes




