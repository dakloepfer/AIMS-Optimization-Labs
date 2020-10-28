import torch

from objective.base import Objective
from utils import accuracy

class SVM(Objective):
    def __init__(self, hparams):
        super(SVM, self).__init__(hparams)
        self._range = torch.arange(hparams.n_classes)[None, :]

    def task_error(self, w, x, y):
        self._validate_inputs(w, x, y)
        # Compute mean misclassification
        scores = torch.mm(x, w)
        error = 1 - accuracy(scores, y)
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
        # Compute objective value
        scores = torch.mm(x, w)

        wyx = scores[torch.arange(x.shape[0]), y]
        delta_ky = torch.ones_like(scores)
        delta_ky[torch.arange(x.shape[0]), y] = 0

        max_arguments = scores + delta_ky - wyx[:, None]

        hinge_loss = torch.mean(torch.max(max_arguments, dim=1)[0])
        obj = hinge_loss + 0.5 * mu * torch.square(torch.norm(w))

        # compute subgradient
        dw = torch.zeros_like(w)

        max_values, max_indices = torch.max(max_arguments, dim=1)

        for sample in range(0, x.shape[0]):
            if max_values[sample] > 0:
                dw[:, max_indices[sample]] += x[sample, :]
                dw[:, y[sample]] -= x[sample, :]

        dw = dw / x.shape[0]
        dw += mu * w

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

        # Compute primal objective value
        scores = torch.mm(x, w)

        wyx = scores[torch.arange(x.shape[0]), y]
        delta_ky = torch.ones_like(scores)
        delta_ky[torch.arange(x.shape[0]), y] = 0
        max_arguments = scores + delta_ky - wyx[:, None]

        hinge_loss = torch.mean(torch.max(max_arguments, dim=1)[0])
        primal = hinge_loss + 0.5 * mu * torch.square(torch.norm(w))

        # Compute w_s
        dw = torch.zeros_like(w)

        max_values, max_indices = torch.max(max_arguments, dim=1)

        for sample in range(0, x.shape[0]):
            if max_values[sample] > 0:
                dw[:, max_indices[sample]] += x[sample, :]
                dw[:, y[sample]] -= x[sample, :]

        w_s = -1/(mu * n_samples) * dw

        # Compute l_s
        l_s = torch.sum(1 - torch.eq(max_indices, y).float()) / n_samples
 
        return {'obj': primal, 'w_s': w_s, 'l_s': l_s}
