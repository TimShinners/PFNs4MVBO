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
        num_numeric = 1 if num_features <= 2 else torch.randint(low=1, high=num_features-1, size=[1]).item()
        num_nominal = num_features - num_numeric
        num_categories = torch.randint(low=2, high=12, size=[num_nominal]).tolist()

    params = []
    # now we define parameters, first we add numeric variables
    for var in range(num_numeric):
        params += [dict(name='numeric' + str(var), type='num', lb=0, ub=1)]

    # now we add nominal categorical variables
    for i in range(num_nominal):
        params += [
            dict(name='nominal' + str(i), type='nominal', lb=0, ub=1,
                 categories=[str(cat) for cat in range(num_categories[i])])
        ]

    # define the search_space and return
    search_space = SearchSpace(params)
    return search_space

# load cocabo_hp datasets so we can sample quickly later on
try:
    cocabo_hp_dataset = []
    for i in range(7):
        cocabo_hp_dataset = [np.load('/mnt/qb/work/eggensperger/eww768/PFNs4MVBO/cocabo_hps/cocabo_normal_5'+str(i)+'.npy')]

except:
    # these files only exist on the cluster
    # so if I'm on laptop I just skip
    pass


def get_cocabo_hps(input_space, params, model=None):

    cocabo_hp_shapes = [model.kernel.outputscale.shape,
                        [1],
                        model.kernel.base_kernel.numeric_kernel.base_kernel.lengthscale.shape,
                        model.kernel.base_kernel.categorical_kernel.base_kernel.lengthscale.shape,
                        model.kernel.base_kernel.numeric_kernel.outputscale.shape,
                        model.kernel.base_kernel.categorical_kernel.outputscale.shape,
                        1]

    if params.get('hp_sampling_method') == 'from_distributions':
        # sample each hp independently from a distribution
        cocabo_hps = [
            Gamma(params['outputscale_concentration'],
                params['outputscale_rate']).sample(cocabo_hp_shapes[0]),
            Beta(params['lamda_shape'],
                params['lamda_shape']).sample(cocabo_hp_shapes[1]),
            Gamma(params['num_lengthscale_concentration'],
                params['num_lengthscale_rate']
            ).sample(cocabo_hp_shapes[2]),
            Gamma(params['cat_lengthscale_concentration'],
                params['cat_lengthscale_rate']
            ).sample(cocabo_hp_shapes[3]),
            Gamma(params['outputscale_concentration'],
                params['outputscale_rate']
            ).sample(cocabo_hp_shapes[4]),
            Gamma(params['outputscale_concentration'],
                params['outputscale_rate']
            ).sample(cocabo_hp_shapes[5]),
            Gamma(0.5, 200).sample()
        ]

        return cocabo_hps

    if params.get('hp_sampling_method') == 'from_dataset':
        # hps are sampled independently from different .npy files
        hps = []
        for i in range(7):
            hps += [np.random.choice(cocabo_hp_dataset[i], size=cocabo_hp_shapes[i])]
        return hps

    if params.get('hp_sampling_method') == 'authentic':
        # in this function we draw random data, fit the model,
        # then use those AUTHENTIC hyperparameters to generate
        # our real data
        n_data = params.get('n_fake_data', 5)
        if n_data == -1:
            # we make the number of fake data points depend on the number of dimensions
            n_data = input_space.transform(input_space.sample()).shape[1]
        else:
            n_data = torch.randint(low=2, high=n_data, size=[1]).item()

        x = input_space.transform(input_space.sample(n_data))

        if params.get('n_fake_data_method', 'normal') == 'normal':
            y = torch.normal(0, 1, [n_data, 1])
        elif params.get('n_fake_data_method', 'normal') == 'cos':
            # cosine method
            coeffs = (7 * torch.rand(x.shape[1])) + 0
            y = torch.cos(coeffs * x).sum(dim=1)
            y = y.unsqueeze(1)

        # fit model
        model_kwargs = {'dtype': torch.float32,
                        'device': 'cpu'}
        model = BoBuilder.get_model(input_space, 'gp_o', **model_kwargs)
        model.fit(x, y)

        # adjust noise
        if isinstance(params['observation_noise'], float): # and np.random.random() < params['observation_noise']:
            # observation_noise = 0
            observation_noise = params['observation_noise'] * model.likelihood.noise
        else:
            observation_noise = model.likelihood.noise

        # record hyperparameters
        cocabo_hps = [
            model.kernel.outputscale,
            model.kernel.base_kernel.lamda,
            model.kernel.base_kernel.numeric_kernel.base_kernel.lengthscale,
            model.kernel.base_kernel.categorical_kernel.base_kernel.lengthscale,
            model.kernel.base_kernel.numeric_kernel.outputscale,
            model.kernel.base_kernel.categorical_kernel.outputscale,
            observation_noise
        ]

        return cocabo_hps

    raise ValueError("params.get('hp_sampling_method') is invalid!")


def get_model(input_space, x, device=default_device, **hps):
    # need to define search space then model
    model_kwargs = {'dtype': torch.float32,
                    'device': 'cpu'}

    model = BoBuilder.get_model(input_space, 'gp_o', **model_kwargs)

    if hps.get('sample_hyperparameters', False):
        try:
            # sample hyperparameters
            cocabo_hps = get_cocabo_hps(input_space, hps, model=model)

            # assign hyperparameters
            model.kernel.outputscale = cocabo_hps[0]
            model.kernel.base_kernel.lamda = cocabo_hps[1]
            model.kernel.base_kernel.numeric_kernel.base_kernel.lengthscale = cocabo_hps[2]
            model.kernel.base_kernel.categorical_kernel.base_kernel.lengthscale = cocabo_hps[3]
            model.kernel.base_kernel.numeric_kernel.outputscale = cocabo_hps[4]
            model.kernel.base_kernel.categorical_kernel.outputscale = cocabo_hps[5]
            model.likelihood.noise = cocabo_hps[6]
        except:
            pass
            # Explanation: if num_features=1, then we must either have categorical or numeric
            # kernel, not mixture. Thus, we run into attribute errors. Instead of doing a
            # bunch of if/then statements I just decided to skip if it throws an error.
            # Perhaps I should change this later

    # gp only gets defined if we call model.fit(), but we have no data to fit to,
    # so we manually define it here so we can use model.sample_y() later
    model.gp = GPyTorchGPModel(x, torch.Tensor().unsqueeze(0), model.kernel, model.likelihood)

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
