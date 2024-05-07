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

# We run BO experiments with many many iterations


def do_deeptime_experiment(OptimizerClass,
                           list_of_tasks,
                           n_init=10,
                           n_iterations=200,
                           n_setups=10,
                           n_intermediate_obs=99,
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
    n_dims = np.array([18]*n_setups).astype(int)
    numerical_dims = np.random.randint(1, n_dims).astype(int)
    categorical_dims = n_dims - numerical_dims

    experiment_data = pd.DataFrame(columns=[
        'optimizer_name',
        'task_name',
        'num_dims',
        'cat_dims',
        'seed',
        'setup_number',
        'nth_guess',
        'time_observe',
        'time_suggest',
        'new_y',
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

            try:
                optimizer = OptimizerClass(search_space=search_space,
                                           n_init=optimizer_kwargs.get('n_init', n_init),
                                           input_constraints=task.input_constraints,
                                           **optimizer_kwargs)
            except:
                # fix stupid argument issue
                optimizer = OptimizerClass(search_space=search_space,
                                           input_constraints=task.input_constraints,
                                           **optimizer_kwargs)

            task.restart()
            search_space = task.get_search_space()
            optimizer.restart()

            # enforce mcbo acq_optimizer params
            if optimizer.name != 'Random Search':
                # set fast acq_optim_params
                optimizer.acq_optimizer.cont_n_iter = 10

                # for interleaved search
                optimizer.acq_optimizer.n_restarts = 2
                optimizer.acq_optimizer.nominal_tol = 100
                optimizer.acq_optimizer.n_iter = 10

                # reset max data set size
                if not isinstance(optimizer, MVPFNOptimizer):
                    optimizer.model.max_training_dataset_size = 10**10

            # initialize starting data
            x_init = search_space.sample(n_init)
            y_init = task(x_init)
            best_y = y_init.min()
            if task_name in ['perm0', 'power_sum']:
                # prevent overflow errors
                y_init = y_init * (10**-40)

            optimizer.initialize(x_init, y_init)

            n_data_count = y_init.shape[0]

            for i in range(n_iterations):
                print(f'setup:{j}, iteration {i}')

                try:
                    # get suggestion
                    suggest_stopwatch.start()
                    x = optimizer.suggest()
                    suggest_stopwatch.stop()

                    y = task(x)

                    best_y = min([best_y, y.min()])

                    # save original y value in case of transform
                    y_raw = y.copy()
                    if task_name in ['perm0', 'power_sum']:
                        # prevent overflow errors
                        y = y * (10 ** -40)

                    # do observation
                    observe_stopwatch.start()
                    optimizer.observe(x, y)
                    observe_stopwatch.stop()

                    # record data
                    experiment_data.loc[len(experiment_data)] = (
                        optimizer.name,
                        task_name,
                        numerical_dims[j],
                        categorical_dims[j],
                        seed,
                        setup_number,
                        n_data_count,
                        observe_stopwatch.get_elapsed_time(),
                        suggest_stopwatch.get_elapsed_time(),
                        y_raw[0, 0],
                        best_y
                    )

                    # n_data_count is how much was observed prior to the suggest/observe
                    n_data_count += y.shape[0]

                    if i != n_iterations - 1:
                        # now we make big observation to get deeper into the "BO run"
                        x = search_space.sample(n_intermediate_obs)
                        y = task(x)
                        if task_name in ['perm0', 'power_sum']:
                            # prevent overflow errors
                            y = y * (10 ** -40)

                        optimizer.observe(x, y)

                        n_data_count += y.shape[0]
                except torch.cuda.OutOfMemoryError:
                    print("here")
                    1/0
                    return experiment_data
            # clear memory
            print('FINISHED RUN')
            del optimizer, task, search_space
            torch.cuda.empty_cache()
            setup_number += 1

    return experiment_data


if __name__ == "__main__":
    # do argparse business
    parser = argparse.ArgumentParser(description='evaluates PFN performance')
    parser.add_argument('--opt_id', '-oi', type=str, help='ID of the optimizer to be evaluated')
    parser.add_argument('--pfn_filename', '-pf', type=str, required=False, default='',
                        help='filename/path of the pfn, if applicable')
    parser.add_argument('--list_of_tasks', '-lt')
    parser.add_argument('--seed', '-s', type=int, required=False, default=6382846, help='random seed')
    args = parser.parse_args()

    # define list of tasks and other variables
    LIST_OF_TASKS = args.list_of_tasks.split(',')
    data_folder = 'deeptime_experiment_results'
    seed = args.seed
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

    # sort out the surrogate arguments if pfn
    if opt_id[0:3] == 'pfn':
        surrogate_components = surrogate_id.split('_')
        assert surrogate_components[0] == 'pfn'
        prior = surrogate_components[1].lower()
        pfn_number = surrogate_components[2]

        MODEL_FILENAME = f'{directory}/trained_models/pfn_{prior}_{pfn_number}.pth'

        model_kwargs = {
            'pfn': MODEL_FILENAME,
            'num_out': 1,
            'dtype': torch.float32,
            'device': device
        }

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
        n_iterations = 30
        n_setups = 1

        if opt_id[0:3] == 'pfn':

            # sort out the alg components
            opt_id_components = opt_id.split('__')
            surrogate_id = opt_id_components[0]
            acq_id = opt_id_components[1]
            acq_optim_id = opt_id_components[2]
            if len(opt_id_components) > 3 and opt_id_components[3] == 'tr':
                tr_id = 'basic'
            else:
                tr_id = None

            optimizer_kwargs = {
                'pfn_file': MODEL_FILENAME,
                'acq_func': acq_id,
                'acq_optim_name': acq_optim_id,
                'tr_id': tr_id,
                'use_pfn_acq_func': True,
                'device': device,
                'fast': True
            }

            results_bo = do_deeptime_experiment(MVPFNOptimizer, LIST_OF_TASKS, n_init=n_init,
                                                n_iterations=n_iterations, n_setups=n_setups,
                                                n_intermediate_obs=99, seed=seed, **optimizer_kwargs)

        elif opt_id.lower() == 'random':
            # do random suggestions
            results_bo = do_deeptime_experiment(RandomSearch, LIST_OF_TASKS, n_init=n_init,
                                                n_iterations=n_iterations, n_setups=n_setups,
                                                n_intermediate_obs=99, seed=seed)

        else:
            # we use mcbo optimizer
            optimizer_kwargs = {'device': device}

            results_bo = do_deeptime_experiment(BO_ALGOS[opt_id].build_bo, LIST_OF_TASKS, n_init=n_init,
                                                n_iterations=n_iterations, n_setups=n_setups,
                                                n_intermediate_obs=99, seed=seed, **optimizer_kwargs)

        torch.cuda.empty_cache()

        # save results in task-specific csvs
        for task_name in LIST_OF_TASKS:
            results_task = results_bo[results_bo['task_name'] == task_name]
            results_task.to_csv(f'{directory}/{data_folder}/BO_deeptime_{seed}_{task_name}_{surrogate_id.lower()}.csv',
                                index=False)

