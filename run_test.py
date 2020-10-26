import unittest
from sys import argv

from tests.test_ridge import *
from tests.test_lasso import *
from tests.test_svm import *
from tests.test_logistic import *
from tests.test_optimizer import *

if __name__ == '__main__':
    unittest.main(argv=argv)
