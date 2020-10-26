import torch

from optim.base import Optimizer, Variable, HParams


class HParamsClosedForm(HParams):
    required = ('n_samples', 'n_features')
    defaults = {'verbose': 1, 'mu': 0}

    def __init__(self, **kwargs):
        super(HParamsClosedForm, self).__init__(kwargs)


class VariablesClosedForm(Variable):
    def init(self):
        # TODO: Compute size

        assert isinstance(size, tuple)
        # Will contain the weights
        self.w = torch.zeros(size, requires_grad=True)


class ClosedForm(Optimizer):
    def create_vars(self):
        return VariablesClosedForm(self.hparams)

    def get_sampler(self, dataset):
        # this sampler yields the entire dataset
        all_indices = list(range(len(dataset)))
        res = dataset[all_indices]
        yield (-1,) + res

    def get_sampler_len(self, dataset):
        return 1

    def _step(self, oracle_info):
        assert "sol" in oracle_info, \
            "The oracle_info should contain the closed form solution in sol"
        assert oracle_info['sol'].size() == self.variables.w.size(), \
            "The optimal solution should be the same size as w"

        # solution given by oracle
        sol = oracle_info['sol']
        # TODO: Update self.variables.w
