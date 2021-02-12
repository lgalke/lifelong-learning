import gc
import numpy as np
import torch as th
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import GraphSAINTEdgeSampler, GraphSAINTRandomWalkSampler
from torch_geometric.data import GraphSAINTNodeSampler, Data


def cpu_fallback(model, x, edge_index):
    device = x.device
    try:
        output = model(x, edge_index)
        did_shift = False
    except RuntimeError:
        print("[CUDA MEMORY EXCEEDED] Shifting data and model to CPU")
        x = x.cpu()
        edge_index = edge_index.cpu()
        model = model.cpu()
        output = model(x, edge_index)

        did_shift = True

    return output, did_shift




def train_saint(model, optimizer, g, feats, labels, mask=None, epochs=1, weights=None, sampling='node', walk_length=2,
                coverage=200, batch_size=1000, n_jobs=1, device=None):

    if mask is not None:
        assert mask.dtype == th.bool, "Mask needs to be dtype bool for GraphSAINT"
        print("GraphSAINT mask size:", mask.size(0))
        use_mask = True
    else:
        use_mask = False

    use_norm = coverage > 0

    if weights is not None:
        raise NotImplementedError("Weights not implemented for GraphSAINT")

    num_nodes = feats.size(0)
    if isinstance(batch_size, float):
        batch_size = int(num_nodes * batch_size)

    data = Data(x=feats, edge_index=g, y=labels, mask=mask)
    sampler_args = {'data': data, 'batch_size': batch_size,
                    'num_workers': 0, 'num_steps': epochs, 'sample_coverage': coverage,
                    'pin_memory': False} # Pin memory to optimize speed
    if sampling == "node":
        sampler = GraphSAINTNodeSampler
    elif sampling == "edge":
        sampler = GraphSAINTEdgeSampler
    elif sampling == "rw":
        sampler = GraphSAINTRandomWalkSampler
        sampler_args['walk_length'] = walk_length
    elif "class_balanced" in sampling:
        from sampler_torch.sampler import ClassBalancedSampler
        sampler = ClassBalancedSampler
        sampler_args.pop('num_workers')
        sampler_args['n_jobs'] = n_jobs
    else:
        raise NotImplementedError(f"\"{sampling}\" is not a supported sampling method for GraphSAINT!")
    model.train()
    loader = sampler(**sampler_args)

    reduction = 'none' if use_norm else 'mean'

    for i, batch in enumerate(loader):
        # mask -> saint sampled subgraph
        if device is not None:
            batch = batch.to(device)

        if use_norm:
            logits = model(batch.x, batch.edge_index, edge_weight=batch.edge_norm)
        else:
            logits = model(batch.x, batch.edge_index)

        if use_mask:
            subg_mask = batch.mask
            loss = F.cross_entropy(logits[subg_mask], batch.y[subg_mask], reduction=reduction)
            if use_norm:
                loss = (loss * batch.node_norm[subg_mask]).sum()
        else:
            loss = F.cross_entropy(logits, batch.y, reduction=reduction)
            if use_norm:
                loss = (loss * batch.node_norm).sum()


        # This is a no-op as loss is already reduced to its mean
        # print("[debug] Loss size pre-sum", loss.size())
        # loss = loss.sum()
        # Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Step {:d} | Loss: {:.4f}".format(i, loss.detach().item()))
        del batch
        gc.collect()
        th.cuda.empty_cache()

    # No mem leaks
    # del data
    del data
    del loader
    gc.collect()
    th.cuda.empty_cache()


def evaluate_saint(model, g, feats, labels, mask=None, compute_loss=True, compute_f1=False):
    model.eval()
    with th.no_grad():
        print("GraphSAINT eval labels size", labels.size())
        logits = model(feats, g)
        print("GraphSAINT eval logits size", logits.size())
        if mask is not None:
            print("GraphSAINT eval mask size", mask.size())
            logits = logits[mask]
            labels = labels[mask]
        print("Logits size post-mask", logits.size())

        if compute_loss:
            loss = F.cross_entropy(logits, labels).item()
        else:
            loss = None

        __max_vals, max_indices = th.max(logits.detach(), 1)
        acc = (max_indices == labels).sum().float() / labels.size(0)
        if compute_f1:
            f1 = f1_score(labels.cpu(), max_indices.cpu(), average="macro")
        else:
            f1 = None
    return acc.item(), f1, loss
