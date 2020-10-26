import unittest
from sys import argv

import numpy as np
import torch

from objective.ridge import Ridge, Ridge_ClosedForm, Ridge_Gradient
from .utils import Container, assert_all_close, assert_all_close_dict


def _init_ridge(cls):
    np.random.seed(1234)
    torch.manual_seed(1234)

    n_features = 3
    n_samples = 5
    mu = 0.02

    cls.hparams = Container(n_features=n_features,
                            n_samples=n_samples,
                            mu=mu)
    cls.w = torch.randn(n_features, 1, requires_grad=True)
    cls.x = torch.randn(n_samples, n_features)
    cls.y = torch.randn(n_samples)


class TestObj_Ridge_ClosedForm(unittest.TestCase):
    def setUp(self):
        _init_ridge(self)
        self.obj = Ridge_ClosedForm(self.hparams)

    def test_error(self):
        error_test = self.obj.task_error(self.w, self.x, self.y)
        error_ref = torch.tensor(1.3251)
        assert_all_close(error_test, error_ref, "task_error returned value")

    def test_oracle(self):
        oracle_info_test = self.obj.oracle(self.w, self.x, self.y)
        oracle_info_ref = {
            'sol': torch.tensor([[-0.2297], [-0.7944], [-0.5806]]),
            'obj': torch.tensor(1.3370)}
        assert_all_close_dict(oracle_info_ref, oracle_info_test, "oracle_info")


class TestObj_Ridge_Gradient(unittest.TestCase):
    def setUp(self):
        _init_ridge(self)
        self.obj = Ridge_Gradient(self.hparams)

    def test_error(self):
        error_test = self.obj.task_error(self.w, self.x, self.y)
        error_ref = torch.tensor(1.3251)
        assert_all_close(error_test, error_ref, "task_error returned value")

    def test_oracle(self):
        oracle_info_test = self.obj.oracle(self.w, self.x, self.y)
        oracle_info_ref = {
            'dw': torch.tensor([[0.7323], [1.4816], [-0.3771]]),
            'obj': torch.tensor(1.3370)}
        assert_all_close_dict(oracle_info_ref, oracle_info_test, "oracle_info")


if __name__ == '__main__':
    unittest.main(argv=argv)
