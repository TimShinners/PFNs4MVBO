import torch
import numpy as np
import pandas as pd
import os
import pfns4bo
from mcbo.trust_region.casmo_tr_manager import CasmopolitanTrManager

import mcbo
from mcbo import task_factory
from mcbo.optimizers.bo_builder import BoBuilder
from mcbo.optimizers.bo_base import BoBase
from mcbo.optimizers.optimizer_base import OptimizerBase
# from mcbo.tasks.synthetic.sfu.utils_sfu import SFU_FUNCTIONS

try:
    # this is a result of weird bugs, one method
    # works on the cluster, one works on my laptop
    from .pfn_model import PFNModel
    from .pfn_acq_func import PFNAcqFunc
    from .mvpfn_optimizer import MVPFNOptimizer
    from .priors import cocabo_prior
except:
    from pfn_model import PFNModel
    from pfn_acq_func import PFNAcqFunc
    from mvpfn_optimizer import MVPFNOptimizer
    from priors import cocabo_prior
from mcbo.utils.experiment_utils import run_experiment
from mcbo.utils.general_utils import create_save_dir, current_time_formatter, set_random_seed
from mcbo.optimizers.bo_builder import BoBuilder, BO_ALGOS
from mcbo.utils.stopwatch import Stopwatch

# first we define functions to make fake data
# then we define functions that make hps
def normal(x):
    # fake data is just drawn from normal distribution
    y = torch.normal(0, 1, [x.shape[0], 1])
    return y


def cosine(x):
    # fake data is generated with a real function, perhaps
    # leading to better hps
    coeffs = (7 * torch.rand(x.shape[1])) + 1
    y = torch.cos(coeffs * x).sum(dim=1)
    y = y.unsqueeze(1)
    return y


def noisy_cosine(x):
    # cosine but we add some noise
    coeffs = (7 * torch.rand(x.shape[1])) + 1
    y = torch.cos(coeffs * x).sum(dim=1)
    y = y + torch.normal(mean=0, std=0.05, size=y.shape)
    y = y.unsqueeze(1)
    return y


def get_cocabo_hps(input_space, y_func, max_points):
    # in this function we draw random data, fit the model,
    # then use those hyperparameters to generate our real data
    # draw data sample
    n_data = torch.randint(low=2, high=max_points, size=[1]).item()
    x = input_space.transform(input_space.sample(n_data))

    y = y_func(x)

    # fit model
    model_kwargs = {'dtype': torch.float32,
                    'device': 'cpu'}
    model = BoBuilder.get_model(input_space, 'gp_o', **model_kwargs)
    model.fit(x, y)

    # record hyperparameters
    hps = [
        model.kernel.outputscale,
        model.kernel.base_kernel.lamda,
        model.kernel.base_kernel.numeric_kernel.base_kernel.lengthscale,
        model.kernel.base_kernel.categorical_kernel.base_kernel.lengthscale,
        model.kernel.base_kernel.numeric_kernel.outputscale,
        model.kernel.base_kernel.categorical_kernel.outputscale,
        model.likelihood.noise
    ]

    return hps


def generate_hps(iterations=100, hp_func=get_cocabo_hps, y_func=normal, max_points=50):
    # we make a for loop and generate many sets of hps
    input_space = cocabo_prior.get_input_space(np.random.randint(2, 18))
    hps = hp_func(input_space, normal, max_points)

    hp_list = [np.zeros(0) for _ in hps]

    for it in range(iterations):
        input_space = cocabo_prior.get_input_space(np.random.randint(2, 18))
        hps = hp_func(input_space, y_func, max_points)
        hp_list = [np.append(hp_list[i], torch.tensor(hps[i]).detach().numpy()) for i, hp in enumerate(hp_list)]

    return hp_list


if __name__ == "__main__":
    hp_list = generate_hps(iterations=10000,
                           hp_func=get_cocabo_hps,
                           y_func=normal,
                           max_points=5)

    for i, hp_set in enumerate(hp_list):
        np.save('cocabo_hps/cocabo_normal_5'+str(i)+'.npy', hp_set)