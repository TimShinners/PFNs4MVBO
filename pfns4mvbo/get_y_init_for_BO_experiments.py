import torch
import numpy as np
import pandas as pd
import scipy
from mcbo import task_factory
from mcbo.optimizers.non_bo.random_search import RandomSearch

try:
    from .pfn_model import PFNModel
    from .pfn_acq_func import PFNAcqFunc
    from .mvpfn_optimizer import MVPFNOptimizer
except:
    from pfn_model import PFNModel
    from pfn_acq_func import PFNAcqFunc
    from mvpfn_optimizer import MVPFNOptimizer
from mcbo.utils.general_utils import set_random_seed
from mcbo.optimizers.bo_builder import BoBuilder, BO_ALGOS
from mcbo.utils.stopwatch import Stopwatch
import argparse
import warnings

# We define functions that run experiments and record data from different optimization runs


def get_y_init_vals_for_BO_experiments(list_of_tasks,
                               n_init=10,
                               n_iterations=200,
                               n_setups=10,
                               seed=3958653,
                               **optimizer_kwargs):
    '''
    for each task, we need to re-initialize the optimizer,
    so instead of taking optimizer as input to this function,
    we take OptimizerClass and optimizer_kwargs
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)
    n_dims = np.random.randint(2, 19, n_setups).astype(int)
    numerical_dims = np.random.randint(1, n_dims).astype(int)
    categorical_dims = n_dims - numerical_dims

    y_init_df = pd.DataFrame(columns=[
        'task_name',
        'num_dims',
        'cat_dims',
        'seed',
        'setup_number',
        'best_y',
    ])

    suggest_stopwatch = Stopwatch()
    observe_stopwatch = Stopwatch()

    setup_number = 0

    for task_name in list_of_tasks:
        np.random.seed(seed)
        torch.manual_seed(seed)
        set_random_seed(seed)
        for j in range(n_setups):
            task_kws = dict(variable_type=['num'] + ['nominal'] * int(categorical_dims[j]),
                            num_dims=[int(numerical_dims[j])] + [1] * int(categorical_dims[j]),
                            num_categories=np.random.randint(2, 10, n_dims[j]).tolist())

            task = task_factory(task_name=task_name, **task_kws)

            search_space = task.get_search_space()



            task.restart()
            search_space = task.get_search_space()

            # initialize starting data
            x_init = search_space.sample(n_init)
            y_init = task(x_init)
            best_y = y_init.min()

            # record data
            y_init_df.loc[len(y_init_df)] = (
                task_name,
                numerical_dims[j],
                categorical_dims[j],
                seed,
                setup_number,
                best_y
            )

            setup_number += 1

    return y_init_df


if __name__ == "__main__":
    # do argparse business
    parser = argparse.ArgumentParser(description='evaluates PFN performance')
    parser.add_argument('--task', '-t', type=str, help='which function to execute')
    parser.add_argument('--opt_id', '-oi', type=str, help='ID of the optimizer to be evaluated')
    parser.add_argument('--pfn_filename', '-pf', type=str, required=False, default='',
                        help='filename/path of the pfn, if applicable')
    parser.add_argument('--list_of_tasks', '-lt')
    parser.add_argument('--seed', '-s', type=int, required=False, default=6382846, help='random seed')
    args = parser.parse_args()

    # define list of tasks and other variables
    LIST_OF_TASKS = ['xgboost_opt', 'schwefel', 'perm0', 'power_sum', 'michalewicz']
    # LIST_OF_TASKS = ['schwefel', 'perm0', 'power_sum', 'michalewicz']
    # LIST_OF_TASKS = args.list_of_tasks.split(',')
    data_folder = 'experiment_results'
    seed = 2468
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # suppress warnings
    warnings.filterwarnings('ignore')

    # set the directory (for cluster)
    directory = 'PFNs4MVBO'

    # task should be regression/overlap/BO/all
    task = 'BO'
    assert task in ['regression', 'overlap', 'BO', 'optimization', 'all']

    # now we sort out the optimizer id
    opt_id = args.opt_id
    opt_id_components = opt_id.split('__')
    surrogate_id = opt_id_components[0]

    # set default model kwargs
    model_kwargs = {'dtype': torch.float32, 'device': device}

    # define mapping from informal -> formal algorithm name
    opt_name_dict = {
        # maps lowercase to actual name used by mcbo
        'cocabo': 'CoCaBO',
        'casmo': 'Casmopolitan',
        'casmopolitan': 'Casmopolitan',
        'bodi': 'BODi',
        'novel': 'novel',
        'mixed': 'mixed'
    }
    if task == 'BO' or task == 'optimization' or task == 'all':
        # settings for BO runs
        n_init = 10
        n_iterations = 200
        n_setups = 15


        # do random suggestions
        results_bo = get_y_init_vals_for_BO_experiments(LIST_OF_TASKS, n_init=n_init,
                                                        n_iterations=n_iterations, n_setups=n_setups, seed=seed)

        # save results in task-specific csvs
        for task_name in LIST_OF_TASKS:
            results_task = results_bo[results_bo['task_name'] == task_name]
            seed_str = '' if seed == 2468 else f'_{str(seed)}'
            results_task.to_csv(f'../experiment_results/BO_{task_name}_y_init{seed_str}.csv',
                                index=False)

