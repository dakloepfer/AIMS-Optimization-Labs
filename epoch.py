import torch

from tqdm import tqdm
from utils import update_metrics


def train(obj, optimizer, dataset, xp, args, epoch):

    for metric in xp.train.metrics():
        metric.reset()
    stats = {}

    for i, x, y in tqdm(optimizer.get_sampler(dataset), desc='Train Epoch',
                        leave=False, total=optimizer.get_sampler_len(dataset)):

        oracle_info = obj.oracle(optimizer.variables.w, x, y)
        oracle_info['i'] = i
        optimizer.step(oracle_info)

        xp.train.error.update(float(obj.task_error(optimizer.variables.w, x, y)))
        xp.train.obj.update(float(oracle_info['obj']))

    xp.train.timer.update()
    for metric in xp.train.metrics():
        metric.log(time=epoch)

    print('\nEpoch: [{0}] (Train) \t'
          '({timer:.2f}s) \t'
          'Obj {obj:.3f}\t'
          'Error {error:.2f}\t'
          .format(int(epoch),
                  timer=xp.train.timer.value,
                  error=xp.train.error.value,
                  obj=xp.train.obj.value,
                  ))


@torch.autograd.no_grad()
def test(obj, optimizer, dataset, xp, args, epoch):
    if dataset.tag == 'val':
        xp_group = xp.val
    else:
        xp_group = xp.test

    for metric in xp_group.metrics():
        metric.reset()


    for idx, x, y in tqdm(optimizer.get_sampler(dataset), leave=False,
                          desc='{} Epoch'.format(dataset.tag.title()),
                          total=optimizer.get_sampler_len(dataset)):
        xp_group.error.update(obj.task_error(optimizer.variables.w, x, y), weighting=x.size(0))

    xp_group.timer.update()
    for metric in xp_group.metrics():
        metric.log(time=xp.epoch.value)

    print('Epoch: [{0}] ({tag})\t'
          '({timer:.2f}s) \t'
          'Obj ----\t'
          'Error {error:.2f}% \t'
          .format(int(xp.epoch.value),
                  tag=dataset.tag.title(),
                  timer=xp_group.timer.value,
                  error=xp_group.error.value))
