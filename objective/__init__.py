from objective.svm import SVM_SubGradient, SVM_ConditionalGradient
from objective.ridge import Ridge_ClosedForm, Ridge_Gradient
from objective.lasso import Lasso_subGradient, SmoothedLasso_Gradient
from objective.logistic import Logistic_Gradient


def get_objective(args, hparams):
    print('Reg Coefficient mu: \t {}'.format(args.mu))

    if args.obj == 'svm':
        if args.opt in ('sgd', 'gd'):
            obj = SVM_SubGradient(hparams)
        elif args.opt == 'bcfw':
            obj = SVM_ConditionalGradient(hparams)
        else:
            raise ValueError

    elif args.obj == 'logistic':
        if args.opt in ('sgd', 'gd'):
            obj = Logistic_Gradient(hparams)
        else:
            raise ValueError

    elif args.obj == 'ridge':
        if args.opt in ('sgd', 'gd'):
            obj = Ridge_Gradient(hparams)
        elif args.opt == 'closed-form':
            obj = Ridge_ClosedForm(hparams)
        else:
            raise ValueError

    elif args.obj == 'lasso':
        if args.opt in ('sgd', 'gd'):
            obj = Lasso_subGradient(hparams)
        else:
            raise ValueError

    elif args.obj == 'smooth-lasso':
        if args.opt in ('sgd', 'gd'):
            obj = SmoothedLasso_Gradient(hparams)
        else:
            raise ValueError

    else:
        raise ValueError('Did not recognize objective {obj}'.format(obj=args.obj))

    return obj
