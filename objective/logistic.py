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
        # TODO: Compute cross entropy prediction error
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
