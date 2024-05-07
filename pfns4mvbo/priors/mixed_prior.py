import time
import random

import torch
from torch.distributions import MultivariateNormal, Beta, Gamma
from torch import nn
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel
import mcbo
from pfns4bo.priors.prior import Batch
from pfns4bo.priors.utils import get_batch_to_dataloader
from mcbo.optimizers.bo_builder import BoBuilder
from mcbo.search_space import SearchSpace
from mcbo.models.gp.exact_gp import GPyTorchGPModel
from pfns4bo.utils import default_device, to_tensor
try:
    from .cocabo_prior import get_model as get_cocabo_model
    from .casmo_prior import get_model as get_casmo_model
    from .bodi_prior import get_model as get_bodi_model
except:
    from cocabo_prior import get_model as get_cocabo_model
    from casmo_prior import get_model as get_casmo_model
    from bodi_prior import get_model as get_bodi_model


def get_input_space(num_features,
                    num_numeric=None,
                    num_nominal=None,
                    num_categories=None):
    # returns a search space given a number of features
    # sample category types
    if num_numeric:
        assert num_features == num_numeric + num_nominal
        assert len(num_categories) == num_nominal
    else:
        num_numeric = 1 if num_features<=2 else torch.randint(low=1, high=num_features-1, size=[1]).item()
        num_nominal = num_features - num_numeric
        num_categories = torch.randint(low=2, high=12, size=[num_nominal]).tolist()

    params = []
    # now we define parameters, first we add numeric variables
    for i in range(num_numeric):
        params += [dict(name='numeric' + str(i), type='num', lb=0, ub=1)]

    # now we add nominal categorical variables
    for i in range(num_nominal):
        params += [
            dict(name='nominal' + str(i), type='nominal', lb=0, ub=1,
                 categories=[str(cat) for cat in range(num_categories[i])])
        ]

    # define the search_space and return
    search_space = SearchSpace(params)
    return search_space


def get_model(input_space, x, device=default_device, **hps):
    get_prior_functions = [get_cocabo_model, get_casmo_model, get_bodi_model]
    prior_hps = [hps['cocabo_hps'], hps['casmo_hps'], hps['bodi_hps']]

    prior_number = np.random.choice(np.arange(len(get_prior_functions)), p=hps['prior_probabilities'])

    model = get_prior_functions[prior_number](input_space, x,
                                              device=default_device,
                                              **prior_hps[prior_number])

    return model


def get_batch(batch_size, seq_len, num_features, device=default_device, hyperparameters=None,
              equidistant_x=False, fix_x=None, **kwargs):

    # define the search space
    input_space = get_input_space(num_features,
                                  num_numeric=kwargs.get('num_numeric', None),
                                  num_nominal=kwargs.get('num_nominal', None),
                                  num_categories=kwargs.get('num_categories', None))

    # get x
    x = input_space.sample(batch_size * seq_len)
    x = input_space.transform(x)
    x_original_shape = x.shape
    x = x.reshape(batch_size, seq_len, num_features)

    # get the model
    model = get_model(input_space, x, **hyperparameters)

    # sample the y's
    sample = torch.zeros([batch_size, seq_len, 1]).to(device)

    for i, batch in enumerate(x):
        sample[i, :, 0] = model.sample_y(batch).flatten()

    # CATEGORICAL ENCODING, we unshape x, transform it, then reshape it
    x = x.view(x_original_shape)
    x = hyperparameters['categorical_encoding_function'](x, input_space)
    x = x.reshape(batch_size, seq_len, num_features)

    # reshape
    x = x.to(torch.float32).to(device)
    x, sample = x.transpose(0, 1), sample.transpose(0, 1)
    target_sample = sample

    return Batch(x=x, y=sample, target_y=target_sample)


DataLoader = get_batch_to_dataloader(get_batch)
