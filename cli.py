import argparse


def parse_command():
    parser = argparse.ArgumentParser()

    _add_dataset_parser(parser)
    _add_optimization_parser(parser)
    _add_obj_parser(parser)
    _add_misc_parser(parser)

    args = parser.parse_args()

    if args.obj in ['ridge', 'lasso', 'smooth-lasso']:
        args.dataset = 'boston' if args.dataset is None else args.dataset
    if args.obj in ['svm', 'logistic']:
        args.dataset = 'mnist' if args.dataset is None else args.dataset

    _validate_args(args)

    return args


def _add_dataset_parser(parser):
    d_parser = parser.add_argument_group(title='Dataset parameters')
    d_parser.add_argument('--dataset', default=None,
                          help='dataset')


def _add_optimization_parser(parser):
    o_parser = parser.add_argument_group(title='Training parameters')
    o_parser.add_argument('--epochs', type=int, default=5,
                          help="number of epochs")
    o_parser.add_argument('--batch-size', type=int, default=256,
                          help="batch size")
    o_parser.add_argument('--init-lr', type=float, default=1e-2,
                          help="initial learning rate")
    o_parser.add_argument('--fix-lr', dest="fix_lr", action="store_true",
                          help="to use a fix learning rate")
    o_parser.add_argument('--opt', type=str)
    o_parser.set_defaults(fix_lr=False)


def _add_obj_parser(parser):
    l_parser = parser.add_argument_group(title='Loss parameters')
    l_parser.add_argument('--mu', type=float, default=1e-3,
                          help="l2 regularization coefficient")
    l_parser.add_argument('--temp', type=float, default=1,
                          help="temperature parameter for smoothed functions")
    l_parser.add_argument('--obj', type=str, default='svm',
                          choices=("svm", "ridge", "lasso", "smooth-lasso", "logistic"))


def _add_misc_parser(parser):
    m_parser = parser.add_argument_group(title='Misc parameters')
    m_parser.add_argument('--seed', type=int, default=None,
                          help="seed for pseudo-randomness")
    m_parser.add_argument('--no-visdom', dest='visdom',
                          action='store_false', help='do not use visdom')
    m_parser.add_argument('--port', type=int, default=8097,
                          help="port for visdom")
    m_parser.add_argument('--xp-name', type=str, default=None,
                          help="name of experiment")
    m_parser.set_defaults(visdom=True)


svm_datasets = ["mnist", "mini-mnist"]
svm_opts = ["bcfw", "gd", "sgd"]
ridge_datasets = ["boston", "california"]
ridge_opts = ["closed-form", "gd", "sgd"]
lasso_datasets = ["boston", "california"]
lasso_opts = ["gd", "sgd"]


def _validate_args(args):
    if args.obj == "svm" or args.obj == "logistic":
        if args.dataset not in svm_datasets:
            raise RuntimeError("{} is not a valid dataset for svm objective".format(args.dataset))
        if args.opt not in svm_opts:
            raise RuntimeError("{} is not a valid opt for svm objective".format(args.opt))
    if args.obj == "ridge":
        if args.dataset not in ridge_datasets:
            raise RuntimeError("{} is not a valid dataset for ridge objective".format(args.dataset))
        if args.opt not in ridge_opts:
            raise RuntimeError("{} is not a valid opt for ridge objective".format(args.opt))
    if args.obj == "lasso" or args.obj == "smooth-lasso":
        if args.dataset not in lasso_datasets:
            raise RuntimeError("{} is not a valid dataset for lasso objective".format(args.dataset))
        if args.opt not in lasso_opts:
            raise RuntimeError("{} is not a valid opt for lasso objective".format(args.opt))
    if args.xp_name is None:
        args.xp_name = "{}_{}_{}".format(args.dataset, args.obj, args.opt)
