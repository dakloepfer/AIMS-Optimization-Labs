from data.datasets import dataset_mnist, dataset_boston, dataset_california, dataset_mini_mnist


def get_datasets(args):

    print('Dataset: \t {}'.format(args.dataset.upper()))

    if args.dataset == 'mnist':
        args.n_features = 784
        args.n_classes = 10
        dataset_train, dataset_val, dataset_test = dataset_mnist()
    elif args.dataset == 'mini-mnist':
        args.n_features = 784
        args.n_classes = 10
        dataset_train, dataset_val, dataset_test = dataset_mini_mnist()
    elif args.dataset == 'boston':
        args.n_features = 13
        args.n_classes = 1
        dataset_train, dataset_val, dataset_test = dataset_boston()
    elif args.dataset == 'california':
        args.n_features = 80000
        args.n_classes = 1
        dataset_train, dataset_val, dataset_test = dataset_california()
    else:
        raise NotImplementedError

    args.train_size = len(dataset_train)
    args.val_size = len(dataset_val)
    args.test_size = len(dataset_test)

    return dataset_train, dataset_val, dataset_test
