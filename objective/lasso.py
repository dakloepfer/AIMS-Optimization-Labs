import torch

from objective.base import Objective


class Lasso(Objective):
    def _validate_inputs(self, w, x, y):
        assert w.dim() == 2, \
            "Input w should be 2D"
        assert w.size(1) == 1, \
            "Lasso regression can only perform regression (size 1 output)"
        assert x.dim() == 2, \
            "Input datapoint should be 2D"
        assert y.dim() == 1, \
            "Input label should be 1D"
        assert x.size(0) == y.size(0), \
            "Input datapoint and label should contain the same number of samples"


class Lasso_subGradient(Lasso):
    def task_error(self, w, x, y):
        self._validate_inputs(w, x, y)

        # Compute mean squared error
        preds = torch.mm(x, w).squeeze()
        error = torch.mean(torch.square(preds - y))
        return error

    def oracle(self, w, x, y):
        self._validate_inputs(w, x, y)
        # regularization hyper-parameter
        mu = self.hparams.mu
        # Compute objective value
        obj = self.task_error(w, x, y) + 0.5 * mu * torch.norm(w, p=1)
        # compute subgradient
        error = torch.mm(x, w).squeeze() - y
        dw = 2 * torch.mean(x.transpose(0, 1) * error, dim=1, keepdim=True) + 0.5 * mu * torch.sign(w)
        return {'obj': obj, 'dw': dw}


class SmoothedLasso_Gradient(Lasso):
    def task_error(self, w, x, y):
        self._validate_inputs(w, x, y)

        # Compute mean squared error
        preds = torch.mm(x, w).squeeze()
        error = torch.mean(torch.square(preds - y))
        return error

    def oracle(self, w, x, y):
        self._validate_inputs(w, x, y)
        # regularization hyper-parameter
        mu = self.hparams.mu
        # temperature parameter
        temp = self.hparams.temp
        # Compute objective value
        obj = self.task_error(w, x, y) + 0.5 * mu * temp * torch.sum(torch.log(2 * torch.cosh(w / temp)))
        # compute gradient
        error = torch.mm(x, w).squeeze() - y
        dw = 2 * torch.mean(x.transpose(0, 1) * error, dim=1, keepdim=True) + 0.5 * mu * torch.tanh(w / temp)
        return {'obj': obj, 'dw': dw}
