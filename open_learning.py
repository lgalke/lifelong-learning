""" Module for Open Learning """
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score


import torch
from torch.nn import Module


class OpenLearning(Module, ABC):
    """ Abstract base class for open world learning """

    @abstractmethod
    def loss(self, logits, labels):
        """ Return loss score to train model """
        raise NotImplementedError("Abstract method called")

    def fit(self, logits, labels):
        """ Hook to learn additional parameters on whole training set """
        return self

    @abstractmethod
    def predict(self, logits, subset=None):
        """ Return most likely classes per instance """
        raise NotImplementedError("Abstract method called")

    @abstractmethod
    def reject(self, logits, subset=None):
        """ Return example-wise mask to emit 1 if reject and 0 otherwise """
        raise NotImplementedError("Abstract method called")

    def forward(self, logits, labels=None, subset=None):

        reject_mask = self.reject(logits, subset=subset)
        predictions = self.predict(logits, subset=subset)
        loss = self.loss(logits, labels) if labels is not None else None

        return reject_mask, predictions, loss


class DeepOpenClassification(OpenLearning):
    """
    Deep Open ClassificatioN: Sigmoidal activation + Threshold based rejection
    Inputs should *not* be activated in any way.
    This module will apply sigmoid activations.
    """
    def __init__(self, threshold: float = 0.5,
                 reduce_risk: bool = False, alpha: float = 3.0,
                 num_classes=None, **kwargs):
        """
        Arguments
        ---------
        threshold: Threshold for class rejection
        alpha: Factor of standard deviation to reduce open space risk
        **kwargs: will be passed to BCEWithLogitsLoss
        """
        super().__init__()
        self.criterion = torch.nn.BCEWithLogitsLoss(**kwargs)
        self.reduce_risk = bool(reduce_risk)
        self.alpha = float(alpha)
        self.threshold = float(threshold)

        self.num_classes = num_classes

        # Minimum threshold if reduce_risk is True,
        # allows to call fit() multiple times
        self.min_threshold = threshold

    def loss(self, logits, labels):
        targets = torch.nn.functional.one_hot(labels,
                                              num_classes=logits.size(1))
        return self.criterion(logits, targets.float())

    def fit(self, logits, labels):
        """ Gaussian fitting of the thresholds per class.
        To be called on the full training set after actual training,
        but before evaluation!
        """
        if not self.reduce_risk:
            print("[DOC/warning] fit() called but reduce_risk is False. Pass.")
            return self

        y = logits.detach().sigmoid()  # [num_examples, num_classes]

        # TODO: extend to online variant by computing *rolling* std. dev.?
        # posterior "probabilities" p(y=l_i | x_j, y_j = li)
        uniq_labels = labels.unique()
        if self.num_classes is None:
            # Infer #classes
            num_classes = len(uniq_labels)
        else:
            num_classes = self.num_classes

        std_per_class = torch.zeros(num_classes)

        for i in uniq_labels:
            # Filter for y_j == li
            y_i = y[labels == i, i]

            # for each existing point,
            # create a mirror point (not a probability),
            # mirrored on the mean of 1
            y_i_mirror = 1 + (1 - y_i)  # [num_examples, num_classes]

            # estimate the standard deviation per class
            # using both existing and the created points
            y_i_all = torch.cat([y_i, y_i_mirror], dim=0)
            # TODO: unbiased SD? orig work did not specify...
            std_i = y_i_all.std(dim=0, unbiased=True)  # scalar

            std_per_class[i] = std_i

        print("SD per class:\n", std_per_class)

        # Set the probability threshold t_i = max(0.5, 1 - alpha * SD_i)
        # Orig paper uses base threshold 0.5,
        # but we use a specified minimum threshold
        thresholds_per_class = (1 - self.alpha * std_per_class).clamp(self.min_threshold)

        self.threshold = thresholds_per_class  # [num_classes]
        print("Updated thresholds:\n", self.threshold)

        return self

    def reject(self, logits, subset=None):
        with torch.no_grad():
            if subset is not None:
                logits = logits[subset]

            y_proba = logits.sigmoid()


            # Dim1 is reduced by 'all' anyways, no mapping back needed
            reject_mask = (y_proba < self.threshold).all(dim=1)
        return reject_mask

    def predict(self, logits, subset=None):
        with torch.no_grad():
            if subset is not None:
                print(f"Reducing view to {len(subset)} known classes")
                logits = logits[subset]

            print("Logits\n", logits)
            y_proba = logits.sigmoid()
            print("Logits after sigmoid\n", y_proba)


            # Basic argmax
            __max_vals, max_indices = torch.max(y_proba, dim=1)
        return max_indices


class OpenMax(OpenLearning):
    pass


##########################
# Module-level functions #
##########################

def add_args(parser):
    parser.add_argument('--open_learning', default=None,
                        help="Method for self detection of unseen classes",
                        choices=["doc"])
    parser.add_argument('--doc_threshold', default=0.5, type=float,
                        help="Threshold for DOC")
    parser.add_argument('--doc_reduce_risk',
                        default=False, action='store_true',
                        help="Reduce Open Space Risk by Gaussian-fitting")
    parser.add_argument('--doc_alpha', default=3.0,
                        help="Alpha for DOC")


def build(args, num_classes=None):
    if args.open_learning == "doc":
        return DeepOpenClassification(threshold=args.doc_threshold,
                                      reduce_risk=args.doc_reduce_risk,
                                      alpha=args.doc_alpha,
                                      num_classes=num_classes)
    elif args.open_learning == "openmax":
        raise NotImplementedError("OpenMax not yet implemented")
    else:
        raise NotImplementedError(f"Unknown key: {args.open_learning}")


def bool2pmone(x):
    """ Converts boolean mask to {-1,1}^N int array """
    x = np.asarray(x, dtype=int)
    return x * 2 - 1


def evaluate(labels, unseen_classes,
             predictions, reject_mask):

    # Shift stuff to CPU
    labels = labels.cpu()
    predictions = predictions.cpu()
    reject_mask = reject_mask.cpu()


    labels = np.asarray(labels)
    unseen = list(unseen_classes)
    predictions = np.asarray(predictions)
    reject_mask = np.asarray(reject_mask)

    print("Labels", labels)
    print("Unseen", unseen)
    true_reject = np.isin(labels, unseen)
    print("True reject", true_reject)
    print("Reject mask", reject_mask)

    print("False in true_reject:", False in true_reject)
    print("True in true_reject:", True in true_reject)
    print("False in reject_mask:", False in reject_mask)
    print("True in reject_mask:", True in reject_mask)

    tp = (reject_mask & true_reject).sum()
    tn = (~reject_mask & ~true_reject).sum()
    fp = (reject_mask & ~true_reject).sum()
    fn = (~reject_mask & true_reject).sum()

    # MCC
    mcc = matthews_corrcoef(bool2pmone(true_reject),
                            bool2pmone(reject_mask))



    # Open F1 Macro
    labels[true_reject] = -100
    print("True lables with -100 for unseen:", labels, labels.shape)
    predictions[reject_mask] = -100
    print("Predictions including rejected:", predictions, predictions.shape)
    f1_macro = f1_score(labels, predictions, average='macro')

    return {'open_mcc': mcc, 'open_f1_macro': f1_macro,
            'open_tp': tp, 'open_tn': tn, 'open_fp': fp, 'open_fn': fn}
