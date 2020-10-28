import torch

from objective.base import Objective


class Ridge(Objective):
    def _validate_inputs(self, w, x, y):
        assert w.dim() == 2, \
                    "Input w should be 2D"
        assert w.size(1) == 1, \
                    "Ridge regression can only perform regression (size 1 output)"
        assert x.dim() == 2, \
                    "Input datapoint should be 2D"
        assert y.dim() == 1, \
                    "Input label should be 1D"
        assert x.size(0) == y.size(0), \
                    "Input datapoint and label should contain the same number of samples"


class Ridge_ClosedForm(Ridge):
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
        obj = self.task_error(w, x, y) + 0.5 * mu * torch.square(torch.norm(w))

        # compute close form solution
        n = x.size(0)
        n_features = x.size(1)
        
        intermediary = torch.inverse(torch.mm(x.transpose(0, 1), x) + 0.5 * n * mu * torch.eye(n_features))
        sol = torch.mv(torch.mm(intermediary, x.transpose(0, 1)), y)
        sol = sol.view(n_features, 1)

        return {'obj': obj, 'sol': sol}


class Ridge_Gradient(Ridge):
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
        obj = self.task_error(w, x, y) + 0.5 * mu * torch.square(torch.norm(w))
        # compute gradient
        preds = torch.mm(x, w).squeeze()
        dw = 2 * torch.mv(x.transpose(0, 1), (preds - y)) / x.shape[0] + mu * w.squeeze()
        dw = dw.view(x.size(1), 1)
        return {'obj': obj, 'dw': dw}
