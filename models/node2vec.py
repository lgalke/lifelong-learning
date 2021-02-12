import torch
import torch_geometric


def add_node2vec_args(parser):
    """ Adds node2vec specific command line arguments to a given parser """
    parser.add_argument('--n2v_walk_length', type=int, default=20,
                        help="Length of node2vec random walks")
    parser.add_argument('--n2v_walks_per_node', type=int, default=10,
                        help="Number of walks per node")
    parser.add_argument('--n2v_num_negative_samples', type=int, default=1,
                        help="Num negative examples per positive example")
    parser.add_argument('--n2v_context_size', type=int, default=10,
                        help="Size of sliding window within random walks")
    parser.add_argument('--n2v_p', type=float, default=1.,
                        help="Node2vec p parameter")
    parser.add_argument('--n2v_q', type=float, default=1.,
                        help="Node2vec q parameter")
    parser.add_argument('--n2v_batch_size', type=int, default=128,
                        help="Node2vec batch size")
    parser.add_argument('--n2v_num_workers', type=int, default=4,
                        help="Node2vec workers (#threads)")

def train_node2vec(model, optimizer, epochs=1,
                   batch_size=128, shuffle=True, num_workers=4):
    assert isinstance(model, torch_geometric.nn.Node2Vec)
    loader = model.loader(batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers)
    device = model.embedding.weight.device # Guess device
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_loss = total_loss / len(loader)
        print("[Node2vec] Epoch {:d} | Loss: {:.4f}".format(epoch + 1, epoch_loss))


@torch.no_grad()
def evaluate_node2vec(model, labels, train_mask_or_vids, test_mask_or_vids):
    assert isinstance(model, torch_geometric.nn.Node2Vec)
    model.eval()
    z = model()
    assert z.size(0) == labels.size(0), "Known embeddings does not match with given labels"
    acc = model.test(z[train_mask_or_vids], labels[train_mask_or_vids],
                     z[test_mask_or_vids], labels[test_mask_or_vids])
    return acc
