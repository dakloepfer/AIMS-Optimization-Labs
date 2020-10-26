import os
import sys
import socket
import torch
import mlogger
import random
import numpy as np


def set_seed(args, print_out=True):
    if args.seed is None:
        np.random.seed(None)
        args.seed = np.random.randint(1e5)
    if print_out:
        print('Seed:\t {}'.format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def get_xp(args, optimizer):

    if args.visdom:
        plotter = mlogger.VisdomPlotter({'env': args.xp_name, 'server': 'http://localhost', 'port': args.port})
    else:
        plotter = None

    xp = mlogger.Container()

    xp.config = mlogger.Config(plotter=plotter, **vars(args))

    xp.epoch = mlogger.metric.Simple()

    xp.train = mlogger.Container()
    xp.train.error = mlogger.metric.Average(plotter=plotter, plot_title="Error", plot_legend="train")
    xp.train.obj = mlogger.metric.Average(plotter=plotter, plot_title="Objective", plot_legend="objective")
    xp.train.timer = mlogger.metric.Timer(plotter=plotter, plot_title="Time", plot_legend='training')

    xp.val = mlogger.Container()
    xp.val.error = mlogger.metric.Average(plotter=plotter, plot_title="Error", plot_legend="validation")
    xp.val.timer = mlogger.metric.Timer(plotter=plotter, plot_title="Time", plot_legend='validation')

    xp.test = mlogger.Container()
    xp.test.error = mlogger.metric.Average(plotter=plotter, plot_title="Error", plot_legend="test")
    xp.test.timer = mlogger.metric.Timer(plotter=plotter, plot_title="Time", plot_legend='test')
    return xp


def print_total_time(xp):
    times = xp.train.timer.state_dict()['history']['times']
    avg_time = np.mean(times)
    total_time = np.sum(times)
    print("\nTotal training time: \t {0:g}s (avg of {1:g}s per epoch)"
          .format(total_time, avg_time))


@torch.autograd.no_grad()
def accuracy(out, targets):
    _, pred = torch.max(out, 1)
    targets = targets.type_as(pred)
    acc = torch.mean(torch.eq(pred, targets).float())
    return acc


def update_metrics(xp, state):
    xp.train.error.update(state['error'], weighting=state['size'])
    xp.train.obj.update(state['obj'], weighting=state['size'])
