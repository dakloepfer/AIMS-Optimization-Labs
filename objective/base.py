
class Objective(object):
    def __init__(self, hparams):
        super(Objective, self).__init__()
        self.hparams = hparams

    def task_error(self, w, x, y):
        raise NotImplementedError

    def oracle(self, w, x, y):
        raise NotImplementedError
