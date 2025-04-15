import numpy as np
import torch
from metric import metric


class ConfusionMatrix(metric.Metric):
    """
    Constructs a confusion matrix for a multi-class classification problems.
    Does not support multi-label, multi-class problems.

    Keyword arguments:
    - num_classes (int): number of classes in the classification problem.
    - normalized (boolean, optional): Determines whether the confusion
    matrix is normalized or not. Default: False.

    Modified from: https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
    """

    def __init__(self, num_classes, normalized=False):
        """
        Initialize the ConfusionMatrix with the number of classes and normalization option.
        """
        super().__init__()

        self.conf = np.ndarray((num_classes, num_classes), dtype=np.int32)
        self.normalized = normalized
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """
        Resets the confusion matrix to zero.
        """
        self.conf.fill(0)

    def add(self, predicted, target):
        """
        Computes the confusion matrix.
        The shape of the confusion matrix is K x K, where K is the number of classes.

        Keyword arguments:
        - predicted (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        predicted scores obtained from the model for N examples and K classes,
        or an N-tensor/array of integer values between 0 and K-1.
        - target (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        ground-truth classes for N examples and K classes, or an N-tensor/array
        of integer values between 0 and K-1.
        """
        # If target and/or predicted are tensors, convert them to numpy arrays
        if torch.is_tensor(predicted):
            predicted = predicted.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], "number of targets and predicted outputs do not match"

        # Additional checks and computations to update the confusion matrix
        ...

    def value(self):
        """
        Returns the current state of the confusion matrix.
        If the matrix is set to be normalized, it returns the normalized confusion matrix.

        Returns:
            Confusion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf
