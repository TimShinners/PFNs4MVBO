import torch
import numpy as np
import pandas as pd
import scipy


def get_validation_data(config, device='cuda:0'):
    """
    For training a new pfn, we need a validation data
    set in order to calculate and record a loss curve.
    This function generates a data set with many
    different hyperparameter samples, dimensionality, etc.

    config is defined in files like train_CASMO.py
    """
    # set seeds
    seed = 12053978
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(42)
    except:
        pass

    # bs is number of different data sets with SAME hyperparameters
    bs = 4
    train_data = []
    test_data = []
    split = 100

    num_hps_list = torch.randint(2, 18, (1, 10))[0]
    n_data_per_set = (split + torch.randint(2, 100, (1, 10))[0]).to(int)
    for n_data in n_data_per_set:
        # batch =
        for num_hps in num_hps_list:
            b = config['priordataloader_class'].get_batch_method(bs, n_data.item(), num_hps.item(), epoch=0,
                                                                 device=device,
                                                                 hyperparameters=
                                                                 {**config['extra_prior_kwargs_dict'][
                                                                     'hyperparameters'],
                                                                  'num_hyperparameter_samples_per_batch': -1, })
            # need to stack on ones w same dims
            train_data += [(b.x[split:], b.y[split:])]
            test_data += [(b.x[:split], b.y[:split])]

    validation_data = {'train': train_data, 'test': test_data}
    return validation_data
















