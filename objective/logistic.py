import torch

from objective.base import Objective


class Logistic_Gradient(Objective):
    def _validate_inputs(self, w, x, y):
        assert w.dim() == 2, \
            "Input w should be 2D"
        assert x.dim() == 2, \
            "Input datapoint should be 2D"
        assert y.dim() == 1, \
            "Input label should be 1D"
        assert x.size(0) == y.size(0), \
            "Input datapoint and label should contain the same number of samples"

    def task_error(self, w, x, y):
        self._validate_inputs(w, x, y)
        # Compute prediction error (fraction of misclassified samples) 
        scores = torch.mm(x, w)
        _, preds = scores.max(dim=1)
        mistakes = (preds != y).float()
        error = mistakes.mean()

        return error

    def oracle(self, w, x, y):
        self._validate_inputs(w, x, y)
        # regularization hyper-parameter
        mu = self.hparams.mu
        # Compute objective value
        obj = torch.mean(torch.log(torch.sum(torch.exp(torch.mm(x, w)), dim=1)) - torch.diag(torch.mm(x, w[:, y])))
        obj += 0.5 * mu * torch.square(torch.norm(w))
        
        # compute gradient
        dw = torch.zeros_like(w)
        for i in range(0, w.shape[1]):
            dwi = torch.mean(x * (torch.exp(torch.mv(x, w[:, i])) / torch.sum(torch.exp(torch.mm(x, w)), dim=1))[:, None] - torch.tensor(i == y).float()[:, None] * x, dim=0)
            dw[:, i] = dwi

        dw += mu * w
        return {'obj': obj, 'dw': dw}
