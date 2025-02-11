# Labs Sessions -- Optimization Course of the AIMS CDT Michealmas 2020

## General Remarks

Most of this code, as well as much of this `README` is taken from a codebase provided for these Labs sessions.

## Installation

This practical will only use python. You should have python 2.7 or 3.x installed on your machine as well as the corresponding pip.

You should download the source code zip file and unzip it in the folder you want. Then you should open a command line and cd into the unzipped folder`.

### (Optional) Use virtualenv

If you want to avoid installing all the practicals library in your global python installation, you can make a virtualenv. This will create a local python install and will allow you to install all libraries locally.

You can install virtualenv if you don't have it by doing `pip install virtualenv`.

To create a new virtual env, you should run: `virtualenv -p /path/to/your/python opt_practical_env` where you replace `/path/to/your/python` with the path to the python executable you want to use (you get this with which python for example if you want to use the default python from your system.

You should then activate the environment every time you want to use this version of python by doing `source opt_practical_env/bin/activate`. You can check by using `which python` that it now uses the virtualenv one instead of the global one.

You can go back to using the global python by simply running `deactivate` that will deactivate the virtualenv.

If you want to delete all the virtualenv and all the installed library, you can simply delete the `opt_practical_env` folder.

### Install dependencies

All dependencies can be install by simply running `pip install -r requirements.txt`. Some are quite large so make sure to do allow some time for this command to run.

If everything ran without errors, you should be able to run `python main.py --help`.

## Code Architecture

The given code base is structured in the following manner:

- `main.py` is the main executable file, all the experiments will be ran by using this file. Run `python main.py --help` to see all arguments it can take.
- `cli.py` contains all the code that handle the command line parsing for the main file.
- `utils.py` contains some utility functions, especially for plots and tests.
- `epoch.py` contains the main training loops functions.
- `run_test.py` is the main test file. See the Running Tests section below for more details.
- `optim/` is the folder containing all the different optimizers, see `Optimizer` below for more details of how it works.
- `objective/` is the folder containing all the different objectives, see `Objective` below for more details of how it works.
- `data/` contains all the functions to load the datasets.
- `tests/` contains all the tests.

## Quick PyTorch introduction

All the code uses the PyTorch library and the full documentation can be found [here](https://pytorch.org/docs/stable/index.html).

This library is quite large and we are only going to use a very small subset of it:

- `torch.Tensor` is a basic Tensor type. A Tensor is a n dimensional array (like a numpy ndarray or a matlab matrix). You can find in the documentation all the functions implemented for these objects. You can read [this](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py) for a quick intro on how to create and use Tensors.
- Dataset utilities to load and get data efficiently. This is already done and you should not need to change it. You can look into the code in the `data/` folder and the `get_sampler` methods of the Optimizer to see how they are used.
- A complete autograd engine, you will see more details at the end of the first practical.

It provides many other things like CUDA support, sparse Tensors, neural network modules and neural network optimization that we are not going to use here but that you can explore.

## Classes To Implement

The two folders where code will need to be modified are the `objective` and `optim` folders. Both folders are structured in a similar way:

- An `__init__.py` file that has a convenient function to get the right objective or optimizer based on a parameter list.
- A `base.py` file that contains the base class definitions (see below).
- One file per subclass that is implemented.

### `Objective`

As can be seen in `objective/base.py`, it is built based on some hyper parameters and should implement two methods:

- `task_error` should return, given weights w, a set of samples x and their corresponding ground truth y, the error made by the classifier in the original task this objective represent. In particular this should not contain any regularization.
- `oracle` is given weights w, a set of samples x and their corresponding ground truth y. It should return a dictionary containing a key `"obj"` that contain the objective value for this Objective. It can contain more information that will be passed to the optimizer, these can include for example gradients.

A Basic implementation would look like:

``` python
class MyObjective(Objective):
      def task_error(self, w, x, y):
          task_error = xxx
          return task error
  
      def oracle(self, w, x, y):
          objective = xxx
          info = xxx
          return {'obj': objective, 'info': info}
```

### `Optimizer`

As can be seen in `optim/base.py`. Along with it, you can find two other class that will be used with the `Optimizer`: Firstly, `HParams` will contain the hyper parameters and should be used by only specifying the required and defaults fields as follow:

``` python
class MyHParams(HParams):
      required = ('param1', 'param2')
      defaults = {'param3': 42}
  
      def __init__(self, **kwargs):
          super(MyHParams, self).__init__(kwargs)
  
  hparams = MyHParams(param1=3, param2="foo")
  print(hparams.param3) # will be 42
```

Secondly, `Variable` will be used to solve all the variables your solver will optimize or needs to track. This can be used for example:

``` python
class MyVariables(Variable):
      def __init__(self, hparams):
          super(MyVariables, self).__init__(hparams)
  
      def init(self):
          self.first_var = torch.rand((3, 5), requires_grad=True) # Will be optimized
          self.step_count = torch.tensor(1., requires_grad=False) # Will be tracked but no optimized
```

Using these, an `Optimizer` needs to implement two methods:

- `create_vars` which should create and set the `.variables` attribute. Most of the time this should be an instance of the `Variable` subclass corresponding to this optimizer.
- `_step` this function is given the oracles info and should update its variables based on them. An optimizer working with gradients will expect the gradient to be given in the oracles info for example.

A Basic implementation would look like:

``` python
class MyOptimizer(Optimizer):
      def create_vars(self):
          return MyVariables(self.hparams)
  
  
      def _step(self, oracle_info):
          self.variables.step_count += 1 # Count the number of steps
          update_first_var = xxx
          self.variables.first_var += update_first_var
```

## Visualization

For visualization of all the training curves, we will use visdom in this practical. It works by having a single server that will provide a webpage for you to see your plots.

To set it up, you can simply open a new terminal and run `python -m visdom.server`. This will start the server on your machine and you can access it at `http://localhost:8097`. You can leave this terminal open in the background during the practical to keep the server running or start it in a background process to keep it up even after this terminal is closed.

By default, all the scripts will send training curves to this server. You can disable these by passing the `--no-visdom` argument to the `main.py` script.

Warning: The plots will disappear when you close the server. If you want to save plots that will be there after the server restarts, you can use the web UI to save some `Environment` using the folder icon next to its name.

## Running Tests

We provide tests for all the elements you will need to implement to make sure that each component works as expected before running the full optimization. You can either run all the tests by running `python run_test.py` or a specific set of tests by calling for example `python run_test.py TestObj_Ridge_ClosedForm`. For each task requiring new code to be implemented, the corresponding test will be given.

If a test fails, that means that your class did not returned the expected value. You should first make sure your maths are correct, that their implementation is correct and potentially print intermediary results to make sure they are what you expect.

## Answers to Questions of Sheet 1

### Question 1.1

The Ridge Regression function is strongly convex because it is a quadratic function.

### Question 1.2

The objective function is smooth, so we can use Gradient Descent to solve it. Since it is a quadratic function, we can also find a closed-form solution to the optimisation problem.
The Hessian of a convex function is positive semi-definite.

### Question 1.3

The gradient is equal to 0 at w_star. See code in ridge.py for solution to this equation.

### Question 1.4

Inverting a matrix is computationally expensive and requires relatively high memory; when these things are a bottleneck and not achieving the best objective function value is tolerable, using a few steps of gradient descent may be preferable.

### Question 1.5

The LASSO regression function is convex but not smooth due to the absolute value in the regularisation term. One can use subgradient descent to optimise it.

### Question 1.6

It takes around 50 epochs.

### Question 1.7

Use log(2*cosh(w/t)) for example, or other functions that one can google.

### Question 1.8

A higher temperature makes the regularisation term more resemble the L2 regularisation (ridge regression) than the L1 regularisation (LASSO). A lower temperature favours sparse weights relatively more.

## Answers to Question Sheet 2

### Question 2.1

We know that log-sum-exp is convex, and composition with an affine map preserves convexity.

### Question 2.2

The gradient w.r.t. w_i is x^T exp(w_i^T x) / sum_k(exp(w_k^T x)) - delta(i, y) x^T.

### Question 2.3

The time complexity of one iteration of gradient descent grows linearly with the number of samples, because the gradients need to be evaluated for every sample in the same way and then combined also in linear time.

### Question 2.4

Taking the maximum of several functions is not in general smooth; in this case there is at least one 'kink', at the point where the hinge loss becomes zero.
A valid sub-gradient of the loss with respect to w_i is s = 0 if L_hinge = 0. If L_hinge > 0, s = -x if i==y, s = +x if i == argmax{w_i x + delta(i, y) - w_y x}, and 0 otherwise.

### Question 2.6 and 2.7

Deriving f(alpha^(t+1)) = f((1-gamma)*alpha^t + gamma s^t) w.r.t. gamma and setting this equal to zero, we find that to minimise f(alpha^(t+1)) we need gamma^star A^T A (s(t) - alpha(t))= b / mu - alpha(t), i.e. we can find gamma^star as the ratio of two vectors. (I am just going to trust that the two vectors are actually parallel). If we find a gamma < 0 or gamma > 1, we need to project to gamma = 0 or 1 respectively, since f(gamma) is convex.

### Question 2.8

Yes, the time complexities make sense.

### Question 2.9

see Algorithm 3 in notes

### Question 2.10

see Algorithm 3 in notes

### Question 2.11

see Algorithm 4 in notes

### Question 2.12

The most expensive operations involve scalar products and other operations on vectors of length Cd, so the time complexity is O(Cd) (for one step). We need to store O(N) copies of such vectors, so the space complexity is O(NCd).

### Question 2.13

The time complexity per step is now O(bCd), because the vectors are longer by a factor b, though for the algorithm to converge we have the same time complexity as in Algorithm 4 because correspondingly fewer steps are needed. The space complexity remains at O(MCd), because we need to store a factor b fewer vectors.
