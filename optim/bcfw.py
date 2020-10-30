import torch
import random
import math

from optim.base import Optimizer, Variable, HParams


class HParamsBCFW(HParams):
    required = ('n_samples', 'batch_size', 'n_features', 'n_classes', 'mu')
    defaults = {'verbose': 1, 'eps': 1e-5}

    def __init__(self, **kwargs):
        super(HParamsBCFW, self).__init__(kwargs)
        self.n_blocks = int(math.ceil(self.n_samples / float(self.batch_size)))


class VariablesBCFW(Variable):
    def __init__(self, hparams):
        """
        Shared Variable.
        """
        super(VariablesBCFW, self).__init__(hparams)

    def init(self):
        self.w = torch.zeros((self.hparams.n_features, self.hparams.n_classes), requires_grad=True)
        self.w_i = torch.zeros((self.hparams.n_blocks, self.hparams.n_features,
                                self.hparams.n_classes))
        self.ll = torch.tensor(0.)
        self.l_i = torch.zeros((self.hparams.n_blocks,))


class BCFW(Optimizer):
    def __init__(self, hparams):
        """
        Shared Variable.
        """
        super(BCFW, self).__init__(hparams)

    def create_vars(self):
        return VariablesBCFW(self.hparams)

    def _step(self, oracle_info):
        # i, w_s, l_s given by oracle
        i = oracle_info['i']
        w_s = oracle_info['w_s']
        l_s = oracle_info['l_s']

        # compute optimal step
        gamma = self._step_size(w_s, l_s, i)

        # perform update
        new_w_i = (1 - gamma) * self.variables.w_i[i] + gamma * w_s
        new_l_i = (1 - gamma) * self.variables.l_i[i] + gamma * l_s
        
        self.variables.w += new_w_i - self.variables.w_i[i]
        self.variables.ll += new_l_i - self.variables.l_i[i]
        self.variables.w_i[i] = new_w_i
        self.variables.l_i[i] = new_l_i

    def _step_size(self, w_s, l_s, i):
        # regularization parameter
        mu = self.hparams.mu

        # compute optimal step size
        if torch.allclose(w_s, self.variables.w_i[i]) and torch.allclose(l_s, self.variables.l_i[i]):
            # return gamma=0 if we would get NaN due to calculating gamma = 0 / 0
            return torch.tensor(0.)

        gamma = - mu * torch.sum(torch.mul(w_s - self.variables.w_i[i], self.variables.w)) + l_s - self.variables.l_i[i]
        gamma = gamma / (mu * torch.square(torch.norm(w_s - self.variables.w_i[i])))
        return gamma.clamp(0., 1.)

    def get_sampler(self, dataset):
        # this sampler shuffles the order of the mini-batches but not
        # which indices each mini-batch contains
        dataset_size = len(dataset)
        batch_size = self.hparams.batch_size
        all_indices = list(range(dataset_size))
        batch_indices = []
        for i in range(int(math.ceil(dataset_size / float(batch_size)))):
            batch_indices.append(
                (i, all_indices[i * batch_size: i * batch_size + batch_size]))
        random.shuffle(batch_indices)
        for i, batch_index in batch_indices:
            x, y = dataset[torch.LongTensor(batch_index)]
            yield (i, x, y.type_as(dataset.target_type))

    def get_sampler_len(self, dataset):
        batch_size = self.hparams.batch_size
        return int(math.ceil(len(dataset) / float(batch_size)))
