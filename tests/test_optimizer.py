import unittest
import itertools
import math
import torch
from torch.utils import data
import numpy as np
import random

from .utils import assert_all_close, assert_all_close_dict
from optim import GD, SGD, BCFW, ClosedForm
from optim import HParamsBCFW, HParamsGD, HParamsClosedForm, HParamsSGD
from objective import Lasso_subGradient, Ridge_ClosedForm, SVM_ConditionalGradient, \
    SVM_SubGradient


class TestOpt_BCFW(unittest.TestCase):
    def setUp(self):
        np.random.seed(1234)
        torch.manual_seed(1234)
        random.seed(1234)

        self.n_features = 3
        self.n_samples = 4
        self.batch_size = 2
        self.n_classes = 3
        self.verbose = 0
        self.mu = 1
        self.n_epochs = 500

        x = torch.randn(self.n_samples, self.n_features)
        label_dist = torch.distributions.categorical.Categorical(torch.ones(self.n_classes))
        y = label_dist.sample((self.n_samples,))
        self.dataset = data.TensorDataset(x, y)
        self.dataset.target_type = torch.LongTensor()
        self.oracle_info = dict(i=1,
                                w_s=torch.randn(self.n_features, self.n_classes),
                                l_s=torch.rand([]))

        self.hparams = HParamsBCFW(n_features=self.n_features,
                                   n_samples=self.n_samples,
                                   batch_size=self.batch_size,
                                   n_classes=self.n_classes,
                                   verbose=self.verbose,
                                   mu=self.mu)

    def test_step(self):
        w_ref = torch.tensor([[-0.0421, 0.0984, -0.0016],
                              [-0.2332, -0.0329, 0.1408],
                              [-0.1098, 0.1021, 0.0310]])

        l_ref = torch.tensor(0.1102)

        w_i_ref = torch.tensor([[[0.0000, 0.0000, 0.0000],
                                 [0.0000, 0.0000, 0.0000],
                                 [0.0000, 0.0000, 0.0000]],
                                [[-0.0421, 0.0984, -0.0016],
                                 [-0.2332, -0.0329, 0.1408],
                                 [-0.1098, 0.1021, 0.0310]]])

        l_i_ref = torch.tensor([0., 0.1102])

        opt = BCFW(self.hparams)
        opt.step(self.oracle_info)

        assert_all_close(opt.variables.w, w_ref, "variable w_ref")
        assert_all_close(opt.variables.w_i, w_i_ref, "variable w_i_ref")
        assert_all_close(opt.variables.ll, l_ref, "variable l_ref")
        assert_all_close(opt.variables.l_i, l_i_ref, "variable l_i_ref")

    def test_convergence(self):

        w_ref = torch.tensor([[0.2977, -0.1279, -0.1698],
                              [0.2401, -0.1616, -0.0784],
                              [0.0239, 0.1476, -0.1715]])

        opt = BCFW(self.hparams)
        obj = SVM_ConditionalGradient(self.hparams)

        for _ in range(self.n_epochs):
            for i, x, y in opt.get_sampler(self.dataset):
                oracle_info = obj.oracle(opt.variables.w, x, y)
                oracle_info['i'] = i
                opt.step(oracle_info)

        w_test = opt.variables.w
        assert_all_close(w_test, w_ref, "final value after training")

    def test_variables_size(self):
        opt = BCFW(self.hparams)
        self.assertEqual(opt.variables.ll.size(),
                         torch.Size(),
                         "size of ll variable")
        self.assertEqual(opt.variables.l_i.size(),
                         torch.Size([self.hparams.n_blocks]),
                         "size of l_i variable")
        self.assertEqual(opt.variables.w.size(),
                         torch.Size([self.hparams.n_features, self.hparams.n_classes]),
                         "size of w variable")
        self.assertEqual(opt.variables.w_i.size(),
                         torch.Size([self.hparams.n_blocks,
                                     self.hparams.n_features,
                                     self.hparams.n_classes]),
                         "size of w_i variable")


class TestOpt_GD(unittest.TestCase):
    def setUp(self):
        np.random.seed(1234)
        torch.manual_seed(1234)
        random.seed(1234)

        self.n_features = 3
        self.n_samples = 4
        self.n_classes = 1
        self.verbose = 0
        self.init_lr = 1e-2
        self.fix_lr = True
        self.mu = 10
        self.n_epochs = 500

        x = torch.randn(self.n_samples, self.n_features)
        y = torch.randn(self.n_samples)
        self.dataset = data.TensorDataset(x, y)
        self.dataset.target_type = torch.FloatTensor()
        self.oracle_info = dict(dw=torch.randn(self.n_features, 1, requires_grad=True))

        self.hparams = HParamsGD(n_features=self.n_features,
                                 n_samples=self.n_samples,
                                 n_classes=self.n_classes,
                                 verbose=self.verbose,
                                 init_lr=self.init_lr,
                                 fix_lr=self.fix_lr,
                                 mu=self.mu)

    def test_step(self):
        init_lr_vals = [1, 1e-1, 1e-2]
        fix_lr_vals = [True, False]

        w_refs = [
            torch.tensor([[0.6719],
                          [-0.6090],
                          [0.5513]]),
            torch.tensor([[0.7289],
                          [0.0941],
                          [0.7109]]),
            torch.tensor([[0.4451],
                          [0.0045],
                          [0.8849]]),
            torch.tensor([[0.8170],
                          [0.7325],
                          [0.8992]]),
            torch.tensor([[0.6865],
                          [0.7597],
                          [0.9150]]),
            torch.tensor([[0.4001],
                          [0.1080],
                          [0.2541]])
        ]
        it_refs = [10] * 6
        lr_refs = [1., 1. / math.sqrt(9), 0.1, 0.1 / math.sqrt(9), 0.01, 0.01 / math.sqrt(9)]

        for i, (init_lr, fix_lr) in enumerate(itertools.product(init_lr_vals, fix_lr_vals)):
            hparams = HParamsGD(n_features=self.n_features,
                                n_samples=self.n_samples,
                                n_classes=self.n_classes,
                                verbose=self.verbose,
                                init_lr=init_lr,
                                fix_lr=fix_lr)

            opt = GD(hparams)
            opt.variables.it.fill_(9)
            opt.step(self.oracle_info)
            assert_all_close(opt.variables.w, w_refs[i],
                             "variable w for init_lr={} and fix_lr={}".format(init_lr, fix_lr))
            assert_all_close(opt.variables.lr, lr_refs[i],
                             "variable lr for init_lr={} and fix_lr={}".format(init_lr, fix_lr))
            assert_all_close(opt.variables.it, it_refs[i],
                             "variable it for init_lr={} and fix_lr={}".format(init_lr, fix_lr))

    def test_convergence(self):
        opt = GD(self.hparams)
        obj = Lasso_subGradient(self.hparams)
        w_ref = torch.tensor([[0.0458], [0.0309], [0.0396]])

        for i in range(self.n_epochs):
            for _, x, y in opt.get_sampler(self.dataset):
                oracle_info = obj.oracle(opt.variables.w, x, y)
                opt.step(oracle_info)
        w_test = opt.variables.w
        assert_all_close(w_test, w_ref, "final value after training")

    def test_variables_size(self):
        opt = GD(self.hparams)
        self.assertEqual(opt.variables.w.size(),
                         torch.Size([self.hparams.n_features,
                                     self.hparams.n_classes]), "size of w variable")
        self.assertEqual(opt.variables.lr.size(), torch.Size(), "size of lr variable")
        self.assertEqual(opt.variables.it.size(), torch.Size(), "size of it variable")


class TestOpt_SGD(unittest.TestCase):
    def setUp(self):
        np.random.seed(1234)
        torch.manual_seed(1234)
        random.seed(1234)

        self.n_features = 3
        self.n_samples = 4
        self.batch_size = 2
        self.n_classes = 1
        self.verbose = 0
        self.init_lr = 1e-3
        self.fix_lr = True
        self.mu = 10
        self.n_epochs = 500

        x = torch.randn(self.n_samples, self.n_features)
        y = torch.randn(self.n_samples)
        self.dataset = data.TensorDataset(x, y)
        self.dataset.target_type = torch.FloatTensor()
        self.oracle_info = dict(dw=torch.randn(self.n_features, 1, requires_grad=True))

        self.hparams = HParamsSGD(n_features=self.n_features,
                                  n_samples=self.n_samples,
                                  n_classes=self.n_classes,
                                  mu=self.mu,
                                  verbose=self.verbose,
                                  batch_size=self.batch_size,
                                  init_lr=self.init_lr,
                                  fix_lr=self.fix_lr)

    def test_step(self):
        init_lr_vals = [1, 1e-1, 1e-2]
        fix_lr_vals = [True, False]

        w_refs = [
            torch.tensor([[0.6719],
                          [-0.6090],
                          [0.5513]]),
            torch.tensor([[0.7289],
                          [0.0941],
                          [0.7109]]),
            torch.tensor([[0.4451],
                          [0.0045],
                          [0.8849]]),
            torch.tensor([[0.8170],
                          [0.7325],
                          [0.8992]]),
            torch.tensor([[0.6865],
                          [0.7597],
                          [0.9150]]),
            torch.tensor([[0.4001],
                          [0.1080],
                          [0.2541]])
        ]
        it_refs = [10] * 6
        lr_refs = [1., 1. / math.sqrt(9), 0.1, 0.1 / math.sqrt(9), 0.01, 0.01 / math.sqrt(9)]

        for i, (init_lr, fix_lr) in enumerate(itertools.product(init_lr_vals, fix_lr_vals)):
            hparams = HParamsSGD(n_features=self.n_features,
                                 n_samples=self.n_samples,
                                 n_classes=self.n_classes,
                                 mu=self.mu,
                                 verbose=self.verbose,
                                 batch_size=self.batch_size,
                                 init_lr=init_lr,
                                 fix_lr=fix_lr)

            opt = SGD(hparams)
            opt.variables.it.fill_(9)
            opt.step(self.oracle_info)
            assert_all_close(opt.variables.w, w_refs[i],
                             "variable w for init_lr={} and fix_lr={}".format(init_lr, fix_lr))
            assert_all_close(opt.variables.lr, lr_refs[i],
                             "variable lr for init_lr={} and fix_lr={}".format(init_lr, fix_lr))
            assert_all_close(opt.variables.it, it_refs[i],
                             "variable it for init_lr={} and fix_lr={}".format(init_lr, fix_lr))

    def test_convergence(self):
        opt = SGD(self.hparams)
        obj = Lasso_subGradient(self.hparams)
        w_ref = torch.tensor([[0.0002], [-0.0012], [0.0044]])

        for i in range(self.n_epochs):
            for _, x, y in opt.get_sampler(self.dataset):
                oracle_info = obj.oracle(opt.variables.w, x, y)
                opt.step(oracle_info)
        w_test = opt.variables.w
        assert_all_close(w_test, w_ref, "final value after training")

    def test_variables_size(self):
        opt = SGD(self.hparams)
        self.assertEqual(opt.variables.w.size(),
                         torch.Size([self.hparams.n_features,
                                     self.hparams.n_classes]), "size of w variable")
        self.assertEqual(opt.variables.lr.size(), torch.Size(), "size of lr variable")
        self.assertEqual(opt.variables.it.size(), torch.Size(), "size of it variable")


class TestOpt_ClosedForm(unittest.TestCase):
    def setUp(self):
        np.random.seed(1234)
        torch.manual_seed(1234)
        random.seed(1234)

        self.n_features = 3
        self.n_samples = 4
        self.mu = 10
        self.n_epochs = 10

        self.hparams = HParamsClosedForm(n_features=self.n_features,
                                         n_samples=self.n_samples,
                                         verbose=0,
                                         mu=self.mu)

        x = torch.randn(self.n_samples, self.n_features)
        y = torch.randn(self.n_samples)
        self.dataset = data.TensorDataset(x, y)
        self.dataset.target_type = torch.FloatTensor()

        self.sol = dict(sol=torch.randn((self.n_features, 1), requires_grad=True))

    def test_step(self):
        opt = ClosedForm(self.hparams)
        opt.step(self.sol)
        w_refs = torch.Tensor([[-0.2611], [0.6104], [-0.0098]])
        assert_all_close(opt.variables.w, w_refs, "variable w")

    def test_convergence(self):
        opt = ClosedForm(self.hparams)
        obj = Ridge_ClosedForm(self.hparams)

        w_ref = torch.tensor([[0.0630], [0.0347], [-0.0308]])
        for i in range(self.n_epochs):
            for _, x, y in opt.get_sampler(self.dataset):
                oracle_info = obj.oracle(opt.variables.w, x, y)
                opt.step(oracle_info)
            if i > 0:
                w_test = opt.variables.w
                assert_all_close(w_test, w_ref, "final value after training")

    def test_variables_size(self):
        opt = ClosedForm(self.hparams)
        self.assertEqual(opt.variables.w.size(),
                         torch.Size([self.hparams.n_features, 1]), "size of w variable")

if __name__ == '__main__':
    unittest.main(argv=argv)
