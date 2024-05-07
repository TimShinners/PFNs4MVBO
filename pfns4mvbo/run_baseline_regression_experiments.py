import torch
import numpy as np
import pandas as pd
import scipy
from mcbo import task_factory
from datetime import datetime
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

def normal_normal_overlap(mu_1, var_1, mu_2, var_2):
    # this function calculates the overlap between two normal
    # distributions (0 is bad, 1 is perfect overlap)
    # it makes a hard integration problem, so we just approximate
    # by choosing many points that form tiny buckets, and
    # calculating the pdfs at each bucket and summing the overlap values
    # logits refers to the raw pfn output, mvpfn refers to the optimizer object

    # define buckets that are dense
    ub = torch.max((mu_1 + 4 * torch.sqrt(var_1)).max(), (mu_2 + 4 * torch.sqrt(var_2)).max()).item()
    lb = torch.min((mu_1 - 4 * torch.sqrt(var_1)).min(), (mu_2 - 4 * torch.sqrt(var_2)).min()).item()
    small_buckets = np.linspace(lb, ub, 10000)
    small_bucket_width = small_buckets[1] - small_buckets[0]

    # calculate normal_pdf for each bucket
    normal_pdf_1 = scipy.stats.norm.pdf(small_buckets, loc=mu_1.detach().cpu(),
                                        scale=np.sqrt(var_1.detach().cpu()))
    normal_pdf_2 = scipy.stats.norm.pdf(small_buckets, loc=mu_2.detach().cpu(),
                                        scale=np.sqrt(var_2.detach().cpu()))

    # calculate the overlap as: integrate min(normal_pdf_1, normal_pdf_2)
    overlap = (normal_pdf_1 <= normal_pdf_2) * normal_pdf_1 + (normal_pdf_1 > normal_pdf_2) * normal_pdf_2
    overlap = (overlap * small_bucket_width).sum(axis=1)
    return overlap


def do_regression_evaluation(list_of_tasks, n_setups, seed, n_test_points=1000, **model_kwargs):
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
        Y_MAX = -np.inf
        Y_MIN = np.inf
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

                task.restart()
                search_space = task.get_search_space()
                x_train = search_space.sample(n_dat)
                y_train = task(x_train)
                Y_MAX = max(Y_MAX, y_train.max())
                Y_MIN = min(Y_MIN, y_train.min())
                if task_name in ['perm0', 'power_sum']:
                    # prevent overflow errors
                    y_train = y_train * (10 ** -40)

                x_train = search_space.transform(x_train).to(device)
                y_train = torch.from_numpy(y_train).to(device)

                fit_stopwatch.start()
                mu = y_train.mean()
                sigma = y_train.var()
                fit_stopwatch.stop()

                # sample a lot
                x_test = search_space.sample(n_test_points)
                y_test = task(x_test)
                Y_MAX = max(Y_MAX, y_test.max())
                Y_MIN = min(Y_MIN, y_test.min())
                if task_name in ['perm0', 'power_sum']:
                    # prevent overflow errors
                    y_test = y_test * (10 ** -40)

                # convert to tensor
                x_test = search_space.transform(x_test).to(device)
                y_test = torch.from_numpy(y_test).to(device)

                # make prediction
                predict_stopwatch.start()
                y_hat = torch.tensor([mu] * x_test.shape[0]).unsqueeze(1)
                var_hat = torch.tensor([sigma] * x_test.shape[0]).unsqueeze(1)
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

                # record data
                regression_data.loc[len(regression_data)] = (
                    trial_number,
                    setup_number,
                    'dummy_baseline',
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
        print(task_name, Y_MIN, Y_MAX)

    # clear memory
    del task, search_space, x_train, y_train, x_test, y_test

    return regression_data


def compare_pfn_vs_mcbo(list_of_tasks, n_setups, n_test_points, seed, mcbo_name, use_saved_outputs, **model_kwargs):
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
            if j % 10 == 0:
                print(task_name, j, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
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

                pfn_fit_stopwatch.start()
                mu = y_train.mean()
                sigma = y_train.var()
                pfn_fit_stopwatch.stop()

                pfn_predict_stopwatch.start()
                y_hat = torch.tensor([mu] * x_test.shape[0]).unsqueeze(1)
                var_hat = torch.tensor([sigma] * x_test.shape[0]).unsqueeze(1)
                pfn_predict_stopwatch.stop()

                overlap = normal_normal_overlap(mu_mcbo, var_mcbo, y_hat, var_hat).mean()

                # record data
                regression_data.loc[len(regression_data)] = (
                    trial_number,
                    setup_number,
                    f'dummy_baseline_{mcbo_name}',
                    task_name,
                    numerical_dims[j],
                    categorical_dims[j],
                    seed,
                    n_dat,
                    pfn_fit_stopwatch.get_elapsed_time(),
                    pfn_predict_stopwatch.get_elapsed_time(),
                    mcbo_fit_time,
                    mcbo_predict_time,
                    overlap
                )
                fileno_counter += 1
                trial_number += 1
            setup_number += 1

    return regression_data


if __name__ == "__main__":
    # do argparse business
    parser = argparse.ArgumentParser(description='evaluates PFN performance')
    parser.add_argument('--task', '-t', type=str, help='which function to execute')
    parser.add_argument('--pfn_filename', '-pf', type=str, required=False, default='',
                        help='filename/path of the pfn, if applicable')
    parser.add_argument('--list_of_tasks', '-lt')
    parser.add_argument('--seed', '-s', type=int, required=False, default=6382846, help='random seed')
    args = parser.parse_args()

    # task should be regression/overlap/BO/all
    task = args.task if args.task is not None else 'regression'
    assert task in ['regression', 'overlap', 'all']

    # define list of tasks and other variables
    # LIST_OF_TASKS = ['xgboost_opt', 'schwefel', 'perm0', 'power_sum', 'michalewicz']
    # LIST_OF_TASKS = ['schwefel', 'perm0', 'power_sum', 'michalewicz']
    LIST_OF_TASKS = args.list_of_tasks.split(',') if args.list_of_tasks is not None else ['schwefel',
                                                                                          'perm0',
                                                                                          'power_sum',
                                                                                          'michalewicz',
                                                                                          'rosenbrock',
                                                                                          'rot_hyp',
                                                                                          'perm']
    LIST_OF_TASKS = ['griewank', 'rosenbrock', 'levy', 'michalewicz', 'schwefel']
    seed = args.seed
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        # we are local
        directory = '..'
        data_folder = f'local_{task}_experiment_results'
        assert task != 'all'
    else:
        # set the directory (for cluster)
        directory = 'PFNs4MVBO'
        data_folder = 'experiment_results'
    device = torch.device(device)

    # suppress warnings
    warnings.filterwarnings('ignore')

    if task == 'regression' or task == 'all':
        n_setups = 100
        n_test_points = 1000
        seed = 2468

        # save results in task-specific csvs
        for task_name in LIST_OF_TASKS:
            # we evaluate mcbo model
            model_kwargs = {'dtype': torch.float32, 'device': torch.device('cpu')}
            regression_results = do_regression_evaluation([task_name], n_setups=n_setups,
                                                          n_test_points=n_test_points, seed=seed,
                                                          **model_kwargs)

            regression_results.to_csv(f'{directory}/{data_folder}/regression_{task_name}_dummy_baseline.csv',
                                index=False)

    if task == 'overlap' or task == 'all':
        for mcbo_name in ['CoCaBO', 'Casmopolitan', 'BODi']:
            # for task_name in ['schwefel', 'perm0', 'power_sum', 'michalewicz']:
            for task_name in LIST_OF_TASKS:
                seed = 2468
                model_kwargs = {'dtype': torch.float32, 'device': torch.device('cpu')}
                # Do overlap with dummy model
                results_task = compare_pfn_vs_mcbo([task_name], n_setups=100, n_test_points=1000, seed=seed,
                                                  mcbo_name=mcbo_name, use_saved_outputs=False,
                                                  **model_kwargs)

                results_task.to_csv(f'../local_overlap_experiment_results/overlap_{task_name}_{mcbo_name}_dummy_baseline.csv', index=False)
                print('SAVED FILE')