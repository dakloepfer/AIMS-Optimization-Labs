import unittest
from sys import argv

import numpy as np
import torch

from objective.logistic import Logistic_Gradient
from .utils import Container, assert_all_close, assert_all_close_dict


class TestObj_Logistic_Gradient(unittest.TestCase):
    def setUp(self):
        np.random.seed(1234)
        torch.manual_seed(1234)

        n_features = 3
        n_samples = 5
        n_classes = 7
        mu = 0.02

        self.hparams = Container(n_classes=n_classes,
                                 n_features=n_features,
                                 n_samples=n_samples,
                                 mu=mu)
        self.w = torch.randn(n_features, n_classes, requires_grad=True)
        self.x = torch.randn(n_samples, n_features)
        self.y = torch.randn(n_samples).long()
        self.obj = Logistic_Gradient(self.hparams)

    def test_error(self):
        error_test = self.obj.task_error(self.w, self.x, self.y)
        error_ref = torch.tensor(1.)
        assert_all_close(error_test, error_ref, "task_error returned value")

    def test_oracle(self):
        oracle_info_test = self.obj.oracle(self.w, self.x, self.y)
        oracle_info_ref = {
            'dw': torch.tensor([[ 0.2578, -0.1417,  0.0046, -0.1236, -0.0180,  0.0249, -0.0273],
                                [-0.3585,  0.1889, -0.0937,  0.0522,  0.0100,  0.1239,  0.0620],
                                [-0.2921,  0.2251, -0.1870,  0.1791,  0.0171,  0.0109, -0.0156]]),
            'obj': torch.tensor(3.1189)}
        assert_all_close_dict(oracle_info_ref, oracle_info_test, "oracle returned info")


if __name__ == '__main__':
    unittest.main(argv=argv)
