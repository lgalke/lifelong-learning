from abc import ABC, abstractmethod
import torch

class OpenLearning(ABC):
    """ Abstract base class for open world learning """

    @abstractmethod
    def loss(self, logits, labels):
        """ Return loss score to train model """
        raise NotImplementedError("Abstract method called")

    @abstractmethod
    def predict(self, logits):
        """ Return most likely classes per instance """
        raise NotImplementedError("Abstract method called")

    @abstractmethod
    def reject(self, logits):
        """ Return example-wise mask to emit 1 if reject and 0 otherwise """
        raise NotImplementedError("Abstract method called")

    def __call__(self, logits, labels=None):

        reject_mask = self.reject(logits)
        predictions = self.predict(logits)
        loss = self.loss(logits, labels) if labels is not None else None

        return reject_mask, predictions, loss


class DOC(OpenLearning):
    """
    Deep Open ClassificatioN: Sigmoidal activation + Threshold based rejection
    Inputs should *not* be activated in any way (neither softmax nor sigmoid)
    """
    def __init__(self, t:float=0.5, alpha=3., **kwargs):
        self.criterion = torch.nn.BCEWithLogitsLoss(**kwargs)
        self.threshold = t
        self.alpha = alpha

    def loss(self, logits, labels):
        return self.criterion(logits, labels)

    def fit(self, logits, labels):
        """ Gaussian fitting of the thresholds per class.
        To be called on the full training set after actual training, but before evaluation
        """
        # TODO: extend to online variant by computing *rolling* std. dev.?

        num_classes = logits.size(1)
        # posterior "probabilities" p(y=l_i | x_j, y_j = li)
        points = logits.detach().sigmoid()  # [num_examples, num_classes]

        # for each existing point,
        # create a mirror point (not a probability),
        # mirrored on the mean of 1
        mirror_points = 1 + ( 1 - y )  # [num_examples, num_classes]

        # estimate the standard deviation per class
        # using both existing and the created points
        all_points = torch.cat(points, mirror_points) # [2*num_examples, num_classes]
        std_per_class = all_points.std(dim=0, unbiased=True)  # [num_classes]
        # TODO: unbiased SD? orig work did not specify...

        # Set the probability threshold t_i = max(0.5, 1 - alpha * SD_i)
        thresholds_per_class = (1 - self.alpha * std_per_class).clamp(0.5)  # [num_classes]
        # TODO: Orig paper uses base threshold 0.5, we could also use a specified minimum threshold

        self.threshold = thresholds_per_class  # [num_classes]

        return self

    def reject(self, logits):
        y_proba = logits.detach().sigmoid()
        reject_mask = (y_proba < self.threshold).all(dim=1)
        return reject_mask

    def predict(self, logits):
        y_proba = logits.detach().sigmoid()
        # Basic argmax
        __max_vals, max_indices = torch.max(y_proba, dim=1)
        return max_indices
