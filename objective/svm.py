import torch

from objective.base import Objective
from utils import accuracy

class SVM(Objective):
    def __init__(self, hparams):
        super(SVM, self).__init__(hparams)
        self._range = torch.arange(hparams.n_classes)[None, :]

    def task_error(self, w, x, y):
        self._validate_inputs(w, x, y)
        # TODO: Compute mean misclassification
        error = None
        return error

    def _validate_inputs(self, w, x, y):
        assert w.dim() == 2, "Input w should be 2D"
        assert x.dim() == 2, "Input datapoint should be 2D"
        assert y.dim() == 1, "Input label should be 1D"
        assert x.size(0) == y.size(0), "Input datapoint and label should contain the same number of samples"


class SVM_SubGradient(SVM):
    def __init__(self, hparams):
        super(SVM_SubGradient, self).__init__(hparams)

    def oracle(self, w, x, y):
        self._validate_inputs(w, x, y)
        # regularization hyper-parameter
        mu = self.hparams.mu
        # TODO: Compute objective value
        obj = None
        # TODO: compute subgradient
        dw = None

        return {'obj': obj, 'dw': dw}


class SVM_ConditionalGradient(SVM):
    def __init__(self, hparams):
        super(SVM_ConditionalGradient, self).__init__(hparams)

    def oracle(self, w, x, y):
        self._validate_inputs(w, x, y)
        # regularization hyper-parameter
        mu = self.hparams.mu
        # number of samples
        n_samples = self.hparams.n_samples
        # size of current mini-batch
        batch_size = x.size(0)
        # TODO: Compute primal objective value
        primal = None
        # TODO: Compute w_s
        w_s = None
        # TODO: Compute l_s
        l_s = None

        return {'obj': primal, 'w_s': w_s, 'l_s': l_s}
