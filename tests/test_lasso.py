import unittest
from sys import argv

import numpy as np
import torch

from objective.lasso import Lasso_subGradient,SmoothedLasso_Gradient
from .utils import Container, assert_all_close, assert_all_close_dict


def _init_lasso(cls):
    np.random.seed(1234)
    torch.manual_seed(1234)

    n_features = 3
    n_samples = 5
    mu = 0.02
    temp = 10

    cls.hparams = Container(n_features=n_features,
                            n_samples=n_samples,
                            mu=mu,
                            temp=temp)
    cls.w = torch.randn(n_features, 1, requires_grad=True)
    cls.x = torch.randn(n_samples, n_features)
    cls.y = torch.randn(n_samples)


class TestObj_Lasso_subGradient(unittest.TestCase):
    def setUp(self):
        _init_lasso(self)
        self.obj = Lasso_subGradient(self.hparams)

    def test_error(self):
        error_test = self.obj.task_error(self.w, self.x, self.y)
        error_ref = torch.tensor(1.3251)
        assert_all_close(error_test, error_ref, "task_error returned value")

    def test_oracle(self):
        cache_test = self.obj.oracle(self.w, self.x, self.y)
        cache_ref = {
            'dw': torch.tensor([[0.7414], [1.4836], [-0.3669]]),
            'obj': torch.tensor(1.3397)}
        assert_all_close_dict(cache_ref, cache_test, "oracle_info")


class TestObj_SmoothedLasso_Gradient(unittest.TestCase):
    def setUp(self):
        _init_lasso(self)
        self.obj = SmoothedLasso_Gradient(self.hparams)

    def test_error(self):
        error_test = self.obj.task_error(self.w, self.x, self.y)
        error_ref = torch.tensor(1.3251)
        assert_all_close(error_test, error_ref, "task_error returned value")

    def test_oracle(self):
        cache_refs = [{
            'dw': torch.tensor([[0.7357], [1.4836], [-0.3669]]),
            'obj': torch.tensor(1.3400)},{
            'dw': torch.tensor([[0.7319], [1.4774], [-0.3645]]),
            'obj': torch.tensor(1.3511)},{
            'dw': torch.tensor([[0.7315], [1.4740], [-0.3579]]),
            'obj': torch.tensor(1.5336)}
        ]
        temps = [0.1, 1, 10]

        for temp, cache_ref in zip(temps, cache_refs):
            self.hparams.temp = temp
            self.obj = SmoothedLasso_Gradient(self.hparams)
            cache_test = self.obj.oracle(self.w, self.x, self.y)
            assert_all_close_dict(cache_ref, cache_test, "oracle_info with parameter temp={}".format(temp))
            if self.w.grad is not None:
                self.w.grad.zero_()


class TestObj_SmoothedLasso_Gradient_lowtemp(unittest.TestCase):
    def setUp(self):
        _init_lasso(self)
        self.obj = SmoothedLasso_Gradient(self.hparams)

    def test_oracle(self):
        cache_refs = [{
            'dw': torch.tensor([[0.7357], [1.4836], [-0.3669]]),
            'obj': torch.tensor(1.3400)},{
            'dw': torch.tensor([[0.7414], [1.4836], [-0.3669]]),
            'obj': torch.tensor(1.3397)},{
            'dw': torch.tensor([[0.7414], [1.4836], [-0.3669]]),
            'obj': torch.tensor(1.3397)},{
            'dw': torch.tensor([[0.7414], [1.4836], [-0.3669]]),
            'obj': torch.tensor(1.3397)},{
            'dw': torch.tensor([[0.7414], [1.4836], [-0.3669]]),
            'obj': torch.tensor(1.3397)}
        ]
        temps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

        for temp, cache_ref in zip(temps, cache_refs):
            self.hparams.temp = temp
            self.obj = SmoothedLasso_Gradient(self.hparams)
            cache_test = self.obj.oracle(self.w, self.x, self.y)
            assert_all_close_dict(cache_ref, cache_test, "oracle_info with parameter temp={}".format(temp))
            if self.w.grad is not None:
                self.w.grad.zero_()


if __name__ == '__main__':
    unittest.main(argv=argv)
