from cli import parse_command
from objective import get_objective
from utils import get_xp, set_seed, print_total_time
from data import get_datasets
from optim import get_optimizer
from epoch import train, test


def main(args):

    set_seed(args)

    dataset_train, dataset_val, dataset_test = get_datasets(args)
    optimizer = get_optimizer(args)
    obj = get_objective(args, optimizer.hparams)
    xp = get_xp(args, optimizer)

    for i in range(args.epochs):
        xp.epoch.update(i)
        train(obj, optimizer, dataset_train, xp, args, i)

        test(obj, optimizer, dataset_val, xp, args, i)

    test(obj, optimizer, dataset_test, xp, args, i)
    print_total_time(xp)


if __name__ == '__main__':
    args = parse_command()
    main(args)
