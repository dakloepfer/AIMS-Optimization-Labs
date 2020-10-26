from optim.bcfw import BCFW, HParamsBCFW
from optim.gd import GD, SGD, HParamsGD, HParamsSGD
from optim.closed_form import ClosedForm, HParamsClosedForm


def get_optimizer(args):
    """
    NB: the regularization is included in the controller, hence the zero weight_decay here
    """
    if args.opt == 'bcfw':
        assert not args.fix_lr, "Cannot fix the lr for bcfw"
        hparams = HParamsBCFW(n_samples=args.train_size,
                              batch_size=args.batch_size,
                              n_features=args.n_features,
                              n_classes=args.n_classes,
                              mu=args.mu)
        optimizer = BCFW(hparams)

    elif args.opt == 'gd':
        hparams = HParamsGD(n_samples=args.train_size,
                            n_features=args.n_features,
                            n_classes=args.n_classes,
                            mu=args.mu,
                            init_lr=args.init_lr,
                            fix_lr=args.fix_lr)
        optimizer = GD(hparams)

    elif args.opt == 'sgd':
        hparams = HParamsSGD(n_samples=args.train_size,
                             batch_size=args.batch_size,
                             n_features=args.n_features,
                             n_classes=args.n_classes,
                             mu=args.mu,
                             init_lr=args.init_lr,
                             fix_lr=args.fix_lr)
        optimizer = SGD(hparams)

    elif args.opt == 'closed-form':
        hparams = HParamsClosedForm(n_samples=args.train_size,
                                      n_features=args.n_features,
                                      mu=args.mu)
        optimizer = ClosedForm(hparams)

    else:
        raise ValueError("Dit not recognize optimizer argument {opt}".format(opt=args.opt))

    return optimizer
