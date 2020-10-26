import torch
import numpy as np


class Container:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def assert_all_close(tensor1, tensor2, error_msg, atol=1e-3, rtol=1e-3):
    if torch.is_tensor(tensor1):
        tensor1 = tensor1.detach().numpy()
    if torch.is_tensor(tensor2):
        tensor2 = tensor2.detach().numpy()
    if not np.allclose(tensor1, tensor2, atol=atol, rtol=rtol):
        raise AssertionError("Two results were not equal: {}".format(error_msg))


def assert_all_close_dict(dict1, dict2, error_msg, atol=1e-3, rtol=1e-3):
    assert set(dict1.keys()) == set(dict2.keys())
    for k in dict1:
        key_msg = "key '{}' from ".format(k)
        assert_all_close(dict1[k], dict2[k], key_msg + error_msg, atol, rtol)
