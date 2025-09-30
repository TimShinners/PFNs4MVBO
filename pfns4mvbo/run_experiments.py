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


def normal_logits_overlap(mean, variance, logits, model_pfn):
    # this function calculates the overlap between the pfn and mcbo
    # posterior normal distributions (0 is bad, 1 is perfect overlap)
    # it makes a hard integration problem, so we just approximate
    # by choosing many points that form tiny buckets, and
    # calculating the pdfs at each bucket and summing the overlap values
    # logits refers to the raw pfn output, mvpfn refers to the optimizer object

    # initialize
    probs = torch.softmax(logits, -1).to(device='cpu')
    bucket_borders_pfn = model_pfn.fit_y_to_y(model_pfn.pfn.criterion.borders).to(device='cpu')
    bucket_widths_pfn = bucket_borders_pfn[1:] - bucket_borders_pfn[:-1]

    # define the new precise buckets, using an upper and lower bound
    # make sure we get at least 2 samples per bar in the riemann distribution
    std = torch.sqrt(variance)
    ub = torch.max((mean + 4 * std).max(), bucket_borders_pfn.max() + 0.1).item()
    lb = torch.min((mean - 4 * std).min(), bucket_borders_pfn.min() - 0.1).item()
    density = 10000
    small_buckets = np.linspace(lb, ub, int(density))
    small_bucket_width = small_buckets[1] - small_buckets[0]

    # reformat the probs and get pfn_pdf
    probs = torch.hstack([torch.zeros([probs.shape[0], 1]), probs, torch.zeros([probs.shape[0], 1])])
    pfn_indices = np.digitize(small_buckets, bucket_borders_pfn.detach())
    pfn_pdf = probs[:, pfn_indices].detach().numpy()
    pfn_pdf = pfn_pdf / np.hstack([1, bucket_widths_pfn.numpy(), 1])[pfn_indices]

    # calculate normal_pdf for each bucket
    normal_pdf = scipy.stats.norm.pdf(small_buckets, loc=mean.detach().cpu(), scale=np.sqrt(variance.detach().cpu()))

    # calculate the overlap as integrate min(pfn_pdf, normal_pdf)
    overlap = (pfn_pdf <= normal_pdf) * pfn_pdf + (pfn_pdf > normal_pdf) * normal_pdf
    overlap = (overlap * small_bucket_width).sum(axis=1)
    return overlap


def do_regression_evaluation(ModelClass, list_of_tasks, n_setups, seed, n_test_points=1000, **model_kwargs):
    # first let's do the prior it was trained on
    # then we will do on functions
    device = model_kwargs.get('device', 'cpu')

    def mse(y_test, y_hat):
        sq_err = (y_test-y_hat)**2
        return sq_err.mean()

    def nll(y_test, y_hat, var_hat):
        epsilon = 1e-5
        var_hat[var_hat<epsilon] = epsilon
        NLL = torch.log(var_hat) + (((y_test-y_hat)**2)/var_hat)
        return NLL.mean()

    def esd(p, mu, var):
        # Expected Squared Distance: E_x[(x-p)^2] between
        # normal distribution samples and a given point
        # integral solution from wolfram
        esd = ((mu - p) ** 2) + var
        return esd.mean()

    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)

    # draw the task dimensionality/variable types
    n_dims = np.random.randint(2, 19, n_setups).astype(int)
    numerical_dims = np.random.randint(1, n_dims).astype(int)
    categorical_dims = n_dims - numerical_dims

    n_training_points = np.arange(2, 15)**2

    regression_data = pd.DataFrame(columns=[
        'trial_number',
        'setup_number',
        'model_name',
        'task_name',
        'num_dims',
        'cat_dims',
        'seed',
        'n_training_data',
        'time_fit',
        'time_predict',
        'y_max',
        'y_min',
        'y_mean',
        'mse_raw',
        'mse_scaled',
        'mse_normed',
        'nll_raw',
        'nll_scaled',
        'nll_normed',
        'esd_raw',
        'esd_scaled',
        'esd_normed'
    ])

    # initialize stopwatches
    fit_stopwatch = Stopwatch()
    predict_stopwatch = Stopwatch()

    # initialize trial number
    trial_number = 0
    setup_number = 0

    # now we begin
    for task_name in list_of_tasks:
        np.random.seed(seed)
        torch.manual_seed(seed)
        set_random_seed(seed)
        for j in range(n_setups):
            for n_dat in n_training_points:
                task_kws = dict(variable_type=['num'] + ['nominal'] * int(categorical_dims[j]),
                                num_dims=[int(numerical_dims[j])] + [1] * int(categorical_dims[j]),
                                num_categories=np.random.randint(2, 10, n_dims[j]).tolist())

                if task_name in ['xgboost_opt', 'ackley-53', 'aig_optimization_hyp', 'svm_opt']:
                    # ignore task_kws for real world tasks with predefined dimensionality
                    task = task_factory(task_name=task_name)
                else:
                    task = task_factory(task_name=task_name, **task_kws)

                search_space = task.get_search_space()

                if isinstance(ModelClass, type):
                    model = ModelClass(search_space=search_space, **model_kwargs)
                elif ModelClass.lower() == 'cocabo':
                    model = BO_ALGOS['CoCaBO'].build_bo(search_space=search_space, n_init=1, **model_kwargs).model
                elif ModelClass.lower() == 'casmo' or ModelClass.lower() == 'casmopolitan':
                    model = BO_ALGOS['Casmopolitan'].build_bo(search_space=search_space, n_init=1, **model_kwargs).model
                elif ModelClass.lower() == 'bodi':
                    model = BO_ALGOS['BODi'].build_bo(search_space=search_space, n_init=1, **model_kwargs).model
                else:
                    raise

                task.restart()
                search_space = task.get_search_space()
                x_train = search_space.sample(n_dat)
                y_train = task(x_train)
                if task_name in ['perm0', 'power_sum']:
                    # prevent overflow errors
                    y_train = y_train * (10 ** -40)

                x_train = search_space.transform(x_train).to(device)
                y_train = torch.from_numpy(y_train).to(device)

                fit_stopwatch.start()
                model.fit(x_train, y_train)
                fit_stopwatch.stop()

                # sample a lot
                x_test = search_space.sample(n_test_points)
                y_test = task(x_test)
                if task_name in ['perm0', 'power_sum']:
                    # prevent overflow errors
                    y_test = y_test * (10 ** -40)

                # convert to tensor
                x_test = search_space.transform(x_test).to(device)
                y_test = torch.from_numpy(y_test).to(device)

                # make prediction
                predict_stopwatch.start()
                y_hat, var_hat = model.predict(x_test)
                predict_stopwatch.stop()

                # compute raw losses
                mse_raw = mse(y_test, y_hat).item()
                nll_raw = nll(y_test, y_hat, var_hat).item()
                esd_raw = esd(y_test, y_hat, var_hat).item()

                # prepare to re-normalize/re-scale to 0-1
                y_mean = torch.cat((y_train.flatten(), y_test.flatten()), dim=0).mean().item()
                y_var = torch.cat((y_train.flatten(), y_test.flatten()), dim=0).var().item()
                y_min = torch.tensor([y_train.min(), y_test.min()]).min().item()
                y_max = torch.tensor([y_train.max(), y_test.max()]).max().item()

                # rescale values
                y_hat_scaled = (y_hat - y_min) / (y_max - y_min)
                y_test_scaled = (y_test - y_min) / (y_max - y_min)
                var_hat_scaled = var_hat / (y_max - y_min)

                # calculate scaled losses
                mse_scaled = mse(y_test_scaled, y_hat_scaled).item()
                nll_scaled = nll(y_test_scaled, y_hat_scaled, var_hat_scaled).item()
                esd_scaled = esd(y_test_scaled, y_hat_scaled, var_hat_scaled).item()

                # normalize values
                y_hat_normed = (y_hat - y_mean) / y_var
                y_test_normed = (y_test - y_mean) / y_var
                var_hat_normed = var_hat / y_var

                # calculate normalized losses
                mse_normed = mse(y_test_normed, y_hat_normed).item()
                nll_normed = nll(y_test_normed, y_hat_normed, var_hat_normed).item()
                esd_normed = esd(y_test_normed, y_hat_normed, var_hat_normed).item()

                # fix stupid issue getting model's name
                try:
                    model_name = model.name()
                except TypeError:
                    model_name = model.name

                # record data
                regression_data.loc[len(regression_data)] = (
                    trial_number,
                    setup_number,
                    model_name,
                    task_name,
                    numerical_dims[j],
                    categorical_dims[j],
                    seed,
                    n_dat,
                    fit_stopwatch.get_elapsed_time(),
                    predict_stopwatch.get_elapsed_time(),
                    y_max,
                    y_min,
                    y_mean,
                    mse_raw,
                    mse_scaled,
                    mse_normed,
                    nll_raw,
                    nll_scaled,
                    nll_normed,
                    esd_raw,
                    esd_scaled,
                    esd_normed
                )

                trial_number += 1
            setup_number += 1

    # clear memory
    del model, task, search_space, x_train, y_train, x_test, y_test

    return regression_data


def compare_pfn_vs_mcbo(ModelClass, list_of_tasks, n_setups, n_test_points, seed, mcbo_name, use_saved_outputs, **model_kwargs):
    # first let's do the prior it was trained on
    # then we will do on functions
    device = model_kwargs.get('device', 'cpu')

    def divergence(mu_mcbo, var_mcbo, mu_pfn, var_pfn):
        div = 0.5 * torch.log(var_pfn / var_mcbo) + (var_mcbo + (mu_pfn - mu_mcbo) ** 2) / (2 * var_pfn) - 0.5
        return div.mean().item()

    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)
    n_dims = np.random.randint(2, 19, n_setups).astype(int)
    numerical_dims = np.random.randint(1, n_dims).astype(int)
    categorical_dims = n_dims - numerical_dims

    n_training_points = np.arange(2, 15)**2

    # initialize data set
    regression_data = pd.DataFrame(columns=[
        'trial_number',
        'setup_number',
        'model_name',
        'task_name',
        'num_dims',
        'cat_dims',
        'seed',
        'n_training_data',
        'pfn_fit_time',
        'pfn_predict_time',
        'mcbo_fit_time',
        'mcbo_predict_time',
        'overlap'
    ])

    # initialize stopwatches
    pfn_fit_stopwatch = Stopwatch()
    pfn_predict_stopwatch = Stopwatch()
    mcbo_fit_stopwatch = Stopwatch()
    mcbo_predict_stopwatch = Stopwatch()

    # initialize trial and setup numbers
    trial_number = 0
    setup_number = 0

    # now we begin
    for task_name in list_of_tasks:
        np.random.seed(seed)
        torch.manual_seed(seed)
        set_random_seed(seed)

        fileno_counter = 0
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

                # MCBO
                if not use_saved_outputs:
                    model_mcbo = BO_ALGOS[mcbo_name].build_bo(search_space=search_space, n_init=1, device=device).model

                    mcbo_fit_stopwatch.start()
                    model_mcbo.fit(x_train, y_train)
                    mcbo_fit_stopwatch.stop()

                    mcbo_predict_stopwatch.start()
                    mu_mcbo, var_mcbo = model_mcbo.predict(x_test)
                    mcbo_predict_stopwatch.stop()

                    mcbo_fit_time = mcbo_fit_stopwatch.get_elapsed_time()
                    mcbo_predict_time = mcbo_predict_stopwatch.get_elapsed_time()

                    del model_mcbo

                elif use_saved_outputs:
                    directory = f'PFNs4MVBO/saved_mcbo_experiment_outputs'
                    filename = f'output_{mcbo_name.lower()}_{task_name}_{str(fileno_counter)}.pt'
                    output = torch.load(f'{directory}/{filename}').detach().to(dtype=torch.float64)
                    size = int(output.shape[0] / 5)
                    mu_mcbo = output[:size].unsqueeze(-1)
                    var_mcbo = output[size:2*size].unsqueeze(-1)

                    mcbo_fit_time = output[2*size]
                    mcbo_predict_time = output[3*size]

                    # check seed
                    assert seed == output[4*size]

                # PFN
                model_pfn = ModelClass(search_space=search_space, **model_kwargs)

                pfn_fit_stopwatch.start()
                model_pfn.fit(x_train, y_train)
                pfn_fit_stopwatch.stop()

                pfn_predict_stopwatch.start()
                logits = model_pfn.predict_logits(x_test)
                pfn_predict_stopwatch.stop()

                try:
                    # there's a stupid issue wrt getting model's name
                    model_name = model_pfn.name()
                except TypeError:
                    model_name = model_pfn.name

                overlap = normal_logits_overlap(mu_mcbo, var_mcbo, logits, model_pfn).mean()

                del model_pfn

                # record data
                regression_data.loc[len(regression_data)] = (
                    trial_number,
                    setup_number,
                    model_name,
                    task_name,
                    numerical_dims[j],
                    categorical_dims[j],
                    seed,
                    n_dat,
                    pfn_fit_stopwatch.get_elapsed_time(),
                    pfn_predict_stopwatch.get_elapsed_time(),
                    mcbo_fit_time.cpu(),
                    mcbo_predict_time.cpu(),
                    overlap
                )
                fileno_counter += 1
                trial_number += 1
            setup_number += 1

    return regression_data


def do_optimization_experiment(OptimizerClass,
                               list_of_tasks,
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

            # initialize starting data
            np.random.seed(seed + setup_number)
            torch.manual_seed(seed + setup_number)
            set_random_seed(seed + setup_number)
            x_init = search_space.sample(n_init)
            y_init = task(x_init)
            best_y = y_init.min()
            if task_name in ['perm0', 'power_sum']:
                # prevent overflow errors
                y_init = y_init * (10**-40)

            optimizer.initialize(x_init, y_init)

            experiment_data.loc[len(experiment_data)] = (
                optimizer.name,
                task_name,
                numerical_dims[j],
                categorical_dims[j],
                seed,
                setup_number,
                0,
                observe_stopwatch.get_elapsed_time(),
                suggest_stopwatch.get_elapsed_time(),
                best_y,
                best_y
            )

            for i in range(1, n_iterations + 1):

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
                    i,
                    observe_stopwatch.get_elapsed_time(),
                    suggest_stopwatch.get_elapsed_time(),
                    y_raw[0, 0],
                    best_y
                )

            setup_number += 1

    # clear some memory
    del optimizer, task, search_space

    return experiment_data


if __name__ == "__main__":
    # do argparse business
    parser = argparse.ArgumentParser(description='evaluates PFN performance')
    parser.add_argument('--task', '-t', type=str, help='which function to execute')
    parser.add_argument('--opt_id', '-oi', type=str, help='ID of the optimizer to be evaluated')
    parser.add_argument('--pfn_filename', '-pf', type=str, required=False, default='',
                        help='filename/path of the pfn, if applicable')
    parser.add_argument('--list_of_tasks', '-lt')
    parser.add_argument('--n_BO_setups', '-nbos', type=int, required=False, default=15)
    parser.add_argument('--seed', '-s', type=int, required=False, default=6382846, help='random seed')
    args = parser.parse_args()

    # define list of tasks and other variables
    # LIST_OF_TASKS = ['xgboost_opt', 'schwefel', 'perm0', 'power_sum', 'michalewicz']
    # LIST_OF_TASKS = ['schwefel', 'perm0', 'power_sum', 'michalewicz']
    LIST_OF_TASKS = args.list_of_tasks.split(',')
    data_folder = 'experiment_results'
    seed = args.seed
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print('CUDA NOT AVAILABLE, USING CPU')
        print('device:', device)
    device = torch.device(device)

    # suppress warnings
    warnings.filterwarnings('ignore')

    # set the directory (for cluster)
    directory = 'PFNs4MVBO'

    # task should be regression/overlap/BO/all
    task = args.task
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

    if task == 'regression' or task == 'all':
        n_setups = 100
        n_test_points = 1000

        if surrogate_id[0:3] == 'pfn':
            regression_results = do_regression_evaluation(PFNModel, LIST_OF_TASKS, n_setups=n_setups,
                                                          n_test_points=n_test_points, seed=seed,
                                                          **model_kwargs)

        else:
            # we evaluate mcbo model
            model_kwargs = {'dtype': torch.float32, 'device': torch.device('cpu')}
            regression_results = do_regression_evaluation(opt_id, LIST_OF_TASKS, n_setups=n_setups,
                                                          n_test_points=n_test_points, seed=seed,
                                                          **model_kwargs)

        # save results in task-specific csvs
        for task_name in LIST_OF_TASKS:
            results_task = regression_results[regression_results['task_name'] == task_name]
            results_task.to_csv(f'{directory}/{data_folder}/regression_{task_name}_{surrogate_id.lower()}.csv',
                                index=False)

    if (task == 'overlap' or task == 'all') and surrogate_id[0:3] == 'pfn':
        # prevent overflow errors for perm0 and power_sum
        model_kwargs['dtype'] = torch.float64

        # Do overlap with pfn
        results_div = compare_pfn_vs_mcbo(PFNModel, LIST_OF_TASKS, n_setups=100, n_test_points=1000, seed=seed,
                                          mcbo_name=opt_name_dict[prior.lower()], use_saved_outputs=True,
                                          **model_kwargs)

        # save results in task-specific csvs
        for task_name in LIST_OF_TASKS:
            results_task = results_div[results_div['task_name'] == task_name]
            results_task.to_csv(f'{directory}/{data_folder}/overlap_{task_name}_{surrogate_id.lower()}.csv', index=False)

    if task == 'BO' or task == 'optimization' or task == 'all':
        # settings for BO runs
        n_init = 10
        n_iterations = 200
        n_setups = args.n_BO_setups

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

            results_bo = do_optimization_experiment(MVPFNOptimizer, LIST_OF_TASKS, n_init=n_init,
                                                    n_iterations=n_iterations, n_setups=n_setups, seed=seed,
                                                    **optimizer_kwargs)

        elif opt_id.lower() == 'random':
            # do random suggestions
            results_bo = do_optimization_experiment(RandomSearch, LIST_OF_TASKS, n_init=n_init,
                                                    n_iterations=n_iterations, n_setups=n_setups, seed=seed)

        else:
            # we use mcbo optimizer
            optimizer_kwargs = {'device': device,
                                'obj_dims': None,
                                'out_constr_dims': None,
                                'out_upper_constr_vals': 0,
                                }

            results_bo = do_optimization_experiment(BO_ALGOS[opt_id].build_bo, LIST_OF_TASKS, n_init=n_init,
                                                    n_iterations=n_iterations, n_setups=n_setups, seed=seed,
                                                    **optimizer_kwargs)

        torch.cuda.empty_cache()

        # save results in task-specific csvs
        for task_name in LIST_OF_TASKS:
            results_task = results_bo[results_bo['task_name'] == task_name]
            seed_str = '' if seed == 2468 else f'_{str(seed)}'
            results_task.to_csv(f'{directory}/{data_folder}/BO_{task_name}_{surrogate_id.lower()}{seed_str}.csv',
                                index=False)

