import os
import sklearn
import sklearn.datasets

import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from .utils import random_subsets, Subset


class FlattenedDataset(data.Dataset):
    def __init__(self, dataset):
        super(FlattenedDataset, self).__init__()
        self._dataset = dataset

    def __getitem__(self, i):
        img, label = self._dataset[i]
        # flatten image
        img = img.view(-1)
        return img, label

    def __len__(self):
        return len(self._dataset)


def create_datasets(dataset_train, dataset_val, dataset_test,
                    train_size, val_size, test_size, classification, split=True):

    if split:
        train_indices, val_indices = random_subsets((train_size, val_size),
                                                    len(dataset_train),
                                                    seed=1234)
    else:
        train_indices, = random_subsets((train_size,),
                                        len(dataset_train),
                                        seed=1234)
        val_indices, = random_subsets((val_size,),
                                      len(dataset_val),
                                      seed=1234)

    test_indices, = random_subsets((test_size,),
                                   len(dataset_test),
                                   seed=1234)

    dataset_train = Subset(dataset_train, train_indices)
    dataset_val = Subset(dataset_val, val_indices)
    dataset_test = Subset(dataset_test, test_indices)

    print('Dataset sizes: \t train: {} \t val: {} \t test: {}'
          .format(len(dataset_train), len(dataset_val), len(dataset_test)))

    dataset_train.tag = 'train'
    dataset_val.tag = 'val'
    dataset_test.tag = 'test'

    target_type = torch.LongTensor() if classification else torch.FloatTensor()
    dataset_train.target_type = target_type
    dataset_val.target_type = target_type
    dataset_test.target_type = target_type

    return dataset_train, dataset_val, dataset_test


def dataset_mnist(train_size=50000, val_size=10000, test_size=10000):
    # Data loading code
    normalize = transforms.Normalize(mean=(0.1307,),
                                     std=(0.3081,))

    transform = transforms.Compose([transforms.ToTensor(), normalize])

    # Note: the train-val split will be performed in `create_datasets`
    dataset_train_ = datasets.MNIST(root="data_mnist", train=True, transform=transform, download=True)
    dataset_val_ = datasets.MNIST(root="data_mnist", train=True, transform=transform, download=True)
    dataset_test_ = datasets.MNIST(root="data_mnist", train=False, transform=transform, download=True)

    # we want flattened images
    dataset_train = FlattenedDataset(dataset_train_)
    dataset_val = FlattenedDataset(dataset_val_)
    dataset_test = FlattenedDataset(dataset_test_)

    return create_datasets(dataset_train, dataset_val, dataset_test, train_size,
                           val_size, test_size, True)


def dataset_mini_mnist(train_size=5000, val_size=1000, test_size=1000):
    # Data loading code
    normalize = transforms.Normalize(mean=(0.1307,),
                                     std=(0.3081,))

    transform = transforms.Compose([transforms.ToTensor(), normalize])

    # Note: the train-val split will be performed in `create_datasets`
    dataset_train_ = datasets.MNIST(root="data_mnist", train=True, transform=transform, download=True)
    dataset_val_ = datasets.MNIST(root="data_mnist", train=True, transform=transform, download=True)
    dataset_test_ = datasets.MNIST(root="data_mnist", train=False, transform=transform, download=True)

    # we want flattened images
    dataset_train = FlattenedDataset(dataset_train_)
    dataset_val = FlattenedDataset(dataset_val_)
    dataset_test = FlattenedDataset(dataset_test_)

    return create_datasets(dataset_train, dataset_val, dataset_test, train_size,
                           val_size, test_size, True)


def dataset_boston(train_size=256, val_size=50, test_size=200):

    data_dict = sklearn.datasets.load_boston()
    x = torch.from_numpy(data_dict['data']).float()
    x = x - x.mean(0, keepdim=True)
    x = x / x.std(0, keepdim=True)
    y = torch.from_numpy(data_dict['target']).float()

    # Center and normalize target values
    y -= y.mean()
    y /= y.std()

    dataset = data.TensorDataset(x, y)

    train_indices, val_indices, test_indices = random_subsets((train_size, val_size, test_size),
                                                              len(dataset), seed=1234)

    dataset_train = torch.utils.data.Subset(dataset, train_indices)
    dataset_val = torch.utils.data.Subset(dataset, val_indices)
    dataset_test = torch.utils.data.Subset(dataset, test_indices)

    return create_datasets(dataset_train, dataset_val, dataset_test, train_size,
                           val_size, test_size, False, split=False)


def dataset_california(train_size=10640, val_size=5000, test_size=5000):

    X, y = sklearn.datasets.fetch_california_housing(data_home="data_cali", return_X_y=True)
    x = torch.from_numpy(X).float()
    x = x - x.mean(0, keepdim=True)
    x = x / x.std(0, keepdim=True)
    x = torch.cat([x]*10000, 1)
    y = torch.from_numpy(y).float()

    # Center and normalize target values
    y -= y.mean()
    y /= y.std()

    dataset = data.TensorDataset(x, y)

    train_indices, val_indices, test_indices = random_subsets((train_size, val_size, test_size),
                                                              len(dataset), seed=1234)

    dataset_train = torch.utils.data.Subset(dataset, train_indices)
    dataset_val = torch.utils.data.Subset(dataset, val_indices)
    dataset_test = torch.utils.data.Subset(dataset, test_indices)

    return create_datasets(dataset_train, dataset_val, dataset_test, train_size,
                           val_size, test_size, False, split=False)
