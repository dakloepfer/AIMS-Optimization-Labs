import unittest
from sys import argv

import numpy as np
import torch

from objective.svm import SVM_SubGradient, SVM_ConditionalGradient
from .utils import Container, assert_all_close, assert_all_close_dict


class TestObj_SVM_SubGradient(unittest.TestCase):
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
        self.y = torch.from_numpy(np.random.randint(0, n_classes, size=n_samples))
        self.obj = SVM_SubGradient(self.hparams)

    def test_error(self):
        error_test = self.obj.task_error(self.w, self.x, self.y)
        error_ref = torch.tensor(1.)
        assert_all_close(error_test, error_ref, "task_error returned value")

    def test_oracle(self):
        oracle_info_test = self.obj.oracle(self.w, self.x, self.y)
        oracle_info_ref = {
            'dw': torch.tensor([[-0.0022, -0.2609, 0.0109, 0.0907, -0.0774, 0.2309, -0.0153],
                                [-0.0142, 0.2148, -0.1432, -0.1395, -0.3189, 0.2431, 0.1429],
                                [-0.0133, 0.3670, -0.2574, 0.0139, -0.2911, -0.0592, 0.1777]]),
            'obj': torch.tensor(2.8605)}
        assert_all_close_dict(oracle_info_ref, oracle_info_test, "oracle_info")


class TestObj_SVM_ConditionalGradient(unittest.TestCase):
    def setUp(self):
        np.random.seed(1234)
        torch.manual_seed(1234)

        n_features = 3
        n_samples = 5
        batch_size = 3
        n_classes = 7
        mu = 0.02

        self.hparams = Container(n_classes=n_classes,
                                 n_features=n_features,
                                 batch_size=batch_size,
                                 n_samples=n_samples,
                                 mu=mu)
        self.w = torch.randn(n_features, n_classes, requires_grad=True)
        self.x = torch.randn(batch_size, n_features)
        self.y = torch.from_numpy(np.random.randint(0, n_classes, size=batch_size))
        self.obj = SVM_ConditionalGradient(self.hparams)

    def test_error(self):
        error_test = self.obj.task_error(self.w, self.x, self.y)
        error_ref = torch.tensor(1.)
        assert_all_close(error_test, error_ref, "task_error returned value")

    def test_oracle(self):
        oracle_ref = {'obj': torch.tensor(2.6665),
                      'w_s': torch.tensor([[-0.0000, 11.7741, -0.3821, -5.4156, -0.0000, -6.3585, 0.3821],
                                           [-0.0000, -7.2593, 6.4455, 5.8905, -0.0000, 1.3688, -6.4455],
                                           [-0.0000, -5.3747, 10.3413, 0.7814, -0.0000, 4.5933, -10.3413]]),
                      'l_s': torch.tensor(0.6000)}
        oracle_test = self.obj.oracle(self.w, self.x, self.y)
        assert_all_close_dict(oracle_ref, oracle_test, "oracle_info")


if __name__ == '__main__':
    unittest.main(argv=argv)
