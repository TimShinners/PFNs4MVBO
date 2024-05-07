import pandas as pd
import torch
import numpy as np
from mcbo import task_factory
from mcbo.utils.stopwatch import Stopwatch

try:
    from .pfn_model import PFNModel
    from .pfn_acq_func import PFNAcqFunc
    from .mvpfn_optimizer import MVPFNOptimizer
except:
    from pfn_model import PFNModel
    from pfn_acq_func import PFNAcqFunc
    from mvpfn_optimizer import MVPFNOptimizer
from mcbo.utils.general_utils import create_save_dir, current_time_formatter, set_random_seed
from mcbo.optimizers.bo_builder import BoBuilder, BO_ALGOS
import argparse

# the purpose of this file is to records output for
# many regression tasks and settings. In evaluation.py, we will
# read these results and compare them to the outputs of our pfn


def record_mcbo_output(n_setups, n_test_points, seed, **model_kwargs):
    # first let's do the prior it was trained on
    # then we will do on functions
    device = model_kwargs.get('device', 'cpu')

    list_of_tasks = ['schwefel', 'perm0', 'power_sum', 'michalewicz']

    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)
    n_dims = np.random.randint(2, 19, n_setups).astype(int)
    numerical_dims = np.random.randint(1, n_dims).astype(int)
    categorical_dims = n_dims - numerical_dims

    n_training_points = np.arange(2, 15)**2

    # initialize stopwatches
    mcbo_fit_stopwatch = Stopwatch()
    mcbo_predict_stopwatch = Stopwatch()

    # now we begin
    for task_name in list_of_tasks:
        # reseed for each task
        np.random.seed(seed)
        torch.manual_seed(seed)
        set_random_seed(seed)

        # restart trial count
        trial_number = 0

        for j in range(n_setups):
            for n_dat in n_training_points:
                task_kws = dict(variable_type=['num'] + ['nominal']*int(categorical_dims[j]),
                                num_dims=[int(numerical_dims[j])] + [1]*int(categorical_dims[j]),
                                num_categories=np.random.randint(2, 10, n_dims[j]).tolist())

                task = task_factory(task_name=task_name, **task_kws)

                task.restart()
                search_space = task.get_search_space()
                x_train = search_space.sample(n_dat)
                y_train = task(x_train)
                x_test = search_space.sample(n_test_points)

                # convert to tensor
                x_train = search_space.transform(x_train).to(device)
                y_train = torch.from_numpy(y_train).to(device)
                x_test = search_space.transform(x_test).to(device)

                # for mcbo_name in ['CoCaBO']: # , 'Casmopolitan', 'BODi']:
                for mcbo_name in ['Casmopolitan', 'BODi']:
                    model_mcbo = BO_ALGOS[mcbo_name].build_bo(search_space=search_space, n_init=1, device=device).model

                    mcbo_fit_stopwatch.start()
                    model_mcbo.fit(x_train, y_train)
                    mcbo_fit_stopwatch.stop()

                    mcbo_predict_stopwatch.start()
                    mu_mcbo, var_mcbo = model_mcbo.predict(x_test)
                    mcbo_predict_stopwatch.stop()

                    fit_time = torch.tensor(mu_mcbo.shape[0]*[mcbo_fit_stopwatch.get_elapsed_time()]).detach().to(device=device)
                    predict_time = torch.tensor(mu_mcbo.shape[0] * [mcbo_predict_stopwatch.get_elapsed_time()]).detach().to(device=device)
                    seed_col = torch.tensor(mu_mcbo.shape[0] * [seed]).detach().to(device=device)

                    # record results
                    output = torch.hstack([mu_mcbo.flatten(),
                                           var_mcbo.flatten(),
                                           fit_time.flatten(),
                                           predict_time.flatten(),
                                           seed_col.flatten()])
                    directory = f'PFNs4MVBO/saved_mcbo_experiment_outputs'
                    filename = f'output_{mcbo_name.lower()}_{task_name}_{str(trial_number)}.pt'
                    torch.save(output, f'{directory}/{filename}')

                    # free up memory
                    del model_mcbo
                    del output
                    del mu_mcbo
                    del var_mcbo
                    del fit_time
                    del predict_time
                    torch.cuda.empty_cache()
                del x_train, y_train, x_test


                trial_number += 1

    return


if __name__ == "__main__":
    # do argparse business
    parser = argparse.ArgumentParser(description='This script records cocabo output for later use')
    parser.add_argument('--seed', '-s', type=int, help='random seed')
    args = parser.parse_args()

    seed = args.seed

    # set the device and other parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    device = torch.device(device)

    model_pfn_kwargs = {
        'num_out': 1,
        'dtype': torch.float32,
        'device': device
    }

    record_mcbo_output(n_setups=100, n_test_points=1000, seed=seed, **model_pfn_kwargs)


