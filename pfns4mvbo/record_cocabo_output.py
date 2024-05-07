import torch
import numpy as np
from mcbo import task_factory
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

# the purpose of this file is to records cocabo's output for
# many regression tasks and settings. In evaluation.py, we will
# read these results and compare them to the outputs of our pfn


def record_mcbo_output(mcbo_name, loss_name, n_setups, n_test_points, seed, **model_kwargs):
    # first let's do the prior it was trained on
    # then we will do on functions
    device = model_kwargs.get('device', 'cpu')

    np.random.seed(seed)
    n_dims = np.random.randint(2, 19, n_setups).astype(int)
    numerical_dims = np.random.randint(1, n_dims).astype(int)
    categorical_dims = n_dims - numerical_dims

    n_training_points = np.arange(2, 11)**2

    # just two tasks so we can investigate wrt n_dims and n_data
    list_of_tasks = ['ackley', 'rastrigin', 'trid', 'zakharov', 'dixon_price']

    # initialize trial number
    trial_number = 0
    setup_number = 0

    # now we begin
    for task_name in list_of_tasks:
        for j in range(n_setups):
            for n_dat in n_training_points:
                np.random.seed(seed)
                torch.manual_seed(seed)
                set_random_seed(seed)
                task_kws = dict(variable_type=['num', 'nominal'],
                                num_dims=[int(numerical_dims[j]), int(categorical_dims[j])],
                                num_categories=np.random.randint(2, 10, n_dims[j]).tolist())

                task = task_factory(task_name=task_name, **task_kws)

                set_random_seed(seed)
                task.restart()
                search_space = task.get_search_space()
                x_train = search_space.sample(n_dat)
                y_train = task(x_train)
                x_test = search_space.sample(n_test_points)

                # convert to tensor
                x_train = search_space.transform(x_train).to(device)
                y_train = torch.from_numpy(y_train).to(device)
                x_test = search_space.transform(x_test).to(device)

                # MCBO
                model_mcbo = BO_ALGOS[mcbo_name].build_bo(search_space=search_space, n_init=1, device=device).model

                model_mcbo.fit(x_train, y_train)

                mu_mcbo, var_mcbo = model_mcbo.predict(x_test)

                # record results
                output = torch.hstack([mu_mcbo, var_mcbo])
                torch.save(output, 'PFNs4MVBO/'+mcbo_name.lower()+'_outputs/'+mcbo_name.lower()+'_output_'+str(trial_number)+'.pt')

                trial_number += 1
            setup_number += 1

    return


if __name__ == "__main__":
    # do argparse business
    parser = argparse.ArgumentParser(description='This script records cocabo output for later use')
    parser.add_argument('--mcbo_name', '-m', type=str, help='which mcbo model to evaluate')
    parser.add_argument('--seed', '-s', type=int, help='random seed')
    args = parser.parse_args()

    seed = args.seed
    mcbo_name = args.mcbo_name

    # set the device and other parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # settings for BO runs
    n_init = 10
    n_iterations = 100
    n_setups = 10
    seed = 1235

    model_pfn_kwargs = {
        'num_out': 1,
        'dtype': torch.float32,
        'device': device
    }

    record_mcbo_output(mcbo_name, 'mse', n_setups=100, n_test_points=1000, seed=seed, **model_pfn_kwargs)


