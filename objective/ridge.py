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
        # TODO: Compute mean squared error
        error = None
        return error

    def oracle(self, w, x, y):
        self._validate_inputs(w, x, y)
        # regularization hyper-parameter
        mu = self.hparams.mu
        # TODO: Compute objective value
        obj = None
        # TODO: compute close form solution
        sol = None
        return {'obj': obj, 'sol': sol}


class Ridge_Gradient(Ridge):
    def task_error(self, w, x, y):
        self._validate_inputs(w, x, y)
        # TODO: Compute mean squared error
        error = None
        return error

    def oracle(self, w, x, y):
        self._validate_inputs(w, x, y)
        # regularization hyper-parameter
        mu = self.hparams.mu
        # TODO: Compute objective value
        obj = None
        # TODO: compute gradient
        dw = None
        return {'obj': obj, 'dw': dw}
