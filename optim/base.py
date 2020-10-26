import torch


class Optimizer(object):
    def __init__(self, hparams):
        self.hparams = hparams

        self.variables = self.create_vars()

    def create_vars(self):
        raise NotImplementedError('Variable creation is specific to each optimizer instance')

    def _step(self, oracle_info):
        raise NotImplementedError('Step is specific to each optimizer instance')

    @torch.no_grad()
    def step(self, oracle_info):
        self._step(oracle_info)
        # Only variable w will require gradients
        if hasattr(self.variables, "w") and self.variables.w.grad is not None:
            # Detach from any possible history and reset gradients
            self.variables.w.detach_().requires_grad_(True)
            self.variables.w.grad.zero_()

    def get_sampler(self, dataset):
        raise NotImplementedError('Sampler is specific to each optimizer instance')

    def get_sampler_len(self, dataset):
        raise NotImplementedError('Sampler length is specific to each optimizer instance')


class Variable(object):
    def __init__(self, hparams):
        self.hparams = hparams

        self.init()
        if self.hparams.verbose:
            print(self)

    def init(self):
        raise NotImplementedError('Initialization is specific to each variable instance')

    def __str__(self):
        msg = "Variables: (\n"
        msg += "    - Optimization: \n\t("
        for (k, v) in filter(lambda x: torch.is_tensor(x[1]), self.__dict__.items()):
            size = tuple(v.size()) if v.dim() else 1
            msg += "\n\t * {} \t (size {})".format(k, size)
        msg += "\n\t)\n"
        return msg


class HParams(object):
    required = ()
    defaults = None

    def __init__(self, kwargs):
        super(HParams, self).__init__()
        self.required_keys = set(self.required)
        self.defaults = {} if self.defaults is None else self.defaults.copy()
        allowed_keys = set(self.defaults.keys()).union(self.required_keys)

        # store hyperparameters
        for (k, v) in kwargs.items():
            # check the hyperparameter is valid
            assert k in allowed_keys, "hyper-parameter '{}' not recognized for class {}".format(k, self.__class__.__name__)
            setattr(self, k, v)
            # remove from necessary hyperparameter to store
            if k in self.required_keys:
                self.required_keys.remove(k)
            # remove from default hyperparameters
            if k in self.defaults:
                self.defaults.pop(k)

        # check there is no missing required key
        assert not self.required_keys, "hyper-parameters {} are missing for class {}".format(self.required_keys, self.__class__.__name__)

        # add default values that have not been overwritten
        for (k, v) in self.defaults.items():
            setattr(self, k, v)

        if self.verbose:
            print(self)

    def __str__(self):
        msg = "Hyper-Parameters: (\n"
        msg += "\n\t("
        for (key, value) in self.__dict__.items():
            if key in ('required_keys', 'defaults'):
                continue
            msg += "\n\t * {} \t {}".format(key, value)
        msg += "\n\t)\n"
        msg += ")\n"
        return msg
