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

# first we define a helper function that will compare the
# pfn's output to the cocabo output


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


def do_regression_evaluation(ModelClass, n_setups, seed, n_test_points=1000, **model_kwargs):
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
    n_dims = np.random.randint(2, 19, n_setups).astype(int)
    numerical_dims = np.random.randint(1, n_dims).astype(int)
    categorical_dims = n_dims - numerical_dims

    n_training_points = np.arange(2, 11)**2

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

    # define list of tasks
    list_of_tasks = ['ackley', 'langermann', 'rastrigin', 'trid',
                     'sphere', 'zakharov', 'dixon_price', 'styblinski_tang']

    # initialize stopwatches
    fit_stopwatch = Stopwatch()
    predict_stopwatch = Stopwatch()

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
                task_kws = dict(variable_type=['num'] + ['nominal'] * int(categorical_dims[j]),
                                num_dims=[int(numerical_dims[j])] + [1] * int(categorical_dims[j]),
                                num_categories=np.random.randint(2, 10, n_dims[j]).tolist())

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

                set_random_seed(seed)
                task.restart()
                search_space = task.get_search_space()
                x_train = search_space.sample(n_dat)
                y_train = task(x_train)

                x_train = search_space.transform(x_train).to(device)
                y_train = torch.from_numpy(y_train).to(device)

                fit_stopwatch.start()
                model.fit(x_train, y_train)
                fit_stopwatch.stop()

                # sample a lot
                x_test = search_space.sample(n_test_points)
                y_test = task(x_test)

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
                var_hat_scaled = var_hat / (y_max - y_min) # is this right??

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


def compare_pfn_vs_mcbo(ModelClass, n_setups, n_test_points, seed, mcbo_name='CoCaBO', use_saved_outputs=False, **model_kwargs):
    # first let's do the prior it was trained on
    # then we will do on functions
    device = model_kwargs.get('device', 'cpu')

    def divergence(mu_mcbo, var_mcbo, mu_pfn, var_pfn):
        div = 0.5 * torch.log(var_pfn / var_mcbo) + (var_mcbo + (mu_pfn - mu_mcbo) ** 2) / (2 * var_pfn) - 0.5
        return div.mean().item()

    np.random.seed(seed)
    n_dims = np.random.randint(2, 19, n_setups).astype(int)
    numerical_dims = np.random.randint(1, n_dims).astype(int)
    categorical_dims = n_dims - numerical_dims

    n_training_points = np.arange(2, 11)**2

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
        'div_raw',
        'div_normalized',
        'overlap'
    ])

    # define the list of tasks
    list_of_tasks = ['ackley', 'rastrigin', 'trid', 'zakharov', 'dixon_price']

    # initialize stopwatches
    pfn_fit_stopwatch = Stopwatch()
    pfn_predict_stopwatch = Stopwatch()
    mcbo_fit_stopwatch = Stopwatch()
    mcbo_predict_stopwatch = Stopwatch()

    # initialize trial number
    trial_number = 0
    setup_number = 0

    # now we begin
    for task_name in list_of_tasks:
        # print(task_name, flush=True)
        for j in range(n_setups):
            for n_dat in n_training_points:
                np.random.seed(seed)
                torch.manual_seed(seed)
                set_random_seed(seed)
                task_kws = dict(variable_type=['num'] + ['nominal']*int(categorical_dims[j]),
                                num_dims=[int(numerical_dims[j])] + [1]*int(categorical_dims[j]),
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
                if not use_saved_outputs:
                    model_mcbo = BO_ALGOS[mcbo_name].build_bo(search_space=search_space, n_init=1, device=device).model

                    mcbo_fit_stopwatch.start()
                    model_mcbo.fit(x_train, y_train)
                    mcbo_fit_stopwatch.stop()

                    mcbo_predict_stopwatch.start()
                    mu_mcbo, var_mcbo = model_mcbo.predict(x_test)
                    mcbo_predict_stopwatch.stop()

                    del model_mcbo

                elif use_saved_outputs:
                    if mcbo_name.lower() == 'cocabo':
                        output = torch.load(
                            'PFNs4MVBO/cocabo_outputs/cocabo_output_'+str(trial_number)+'.pt'
                        )
                    elif mcbo_name.lower() == 'casmo' or mcbo_name.lower() == 'casmopolitan':
                        output = torch.load(
                            'PFNs4MVBO/casmopolitan_outputs/casmopolitan_output_'+str(trial_number)+'.pt'
                        )
                    elif mcbo_name.lower() == 'bodi':
                        output = torch.load(
                            'PFNs4MVBO/bodi_outputs/bodi_output_'+str(trial_number)+'.pt'
                        )
                    else:
                        raise ValueError(f'{model_name} outputs not yet recorded/implemented')
                    mu_mcbo = output[:, 0].unsqueeze(-1)
                    var_mcbo = output[:, 1].unsqueeze(-1)

                    # run MCBO clocks to prevent errors (elapsed times should be zero)
                    mcbo_fit_stopwatch.start()
                    mcbo_fit_stopwatch.stop()
                    mcbo_predict_stopwatch.start()
                    mcbo_predict_stopwatch.stop()

                # PFN
                model_pfn = ModelClass(search_space=search_space, **model_kwargs)

                pfn_fit_stopwatch.start()
                model_pfn.fit(x_train, y_train)
                pfn_fit_stopwatch.stop()

                pfn_predict_stopwatch.start()
                mu_pfn, var_pfn = model_pfn.predict(x_test)
                pfn_predict_stopwatch.stop()

                try:
                    # there's a stupid issue wrt getting model's name
                    model_name = model_pfn.name()
                except TypeError:
                    model_name = model_pfn.name

                logits = model_pfn.predict_logits(x_test)

                overlap = normal_logits_overlap(mu_mcbo, var_mcbo, logits, model_pfn).mean()

                del model_pfn

                div_raw = divergence(mu_mcbo, var_mcbo, mu_pfn, var_pfn)

                # normalize with respect to MCBO output
                mu_pfn = mu_pfn - mu_mcbo
                var_pfn = var_pfn / var_mcbo
                mu_mcbo = mu_mcbo - mu_mcbo
                var_mcbo = var_mcbo / var_mcbo

                # calculate divergence
                div_normalized = divergence(mu_mcbo, var_mcbo, mu_pfn, var_pfn)

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
                    mcbo_fit_stopwatch.get_elapsed_time(),
                    mcbo_predict_stopwatch.get_elapsed_time(),
                    div_raw,
                    div_normalized,
                    overlap
                )
                trial_number += 1
            setup_number += 1

    return regression_data


def do_validation_experiment(OptimizerClass,
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
    list_of_tasks = ['ackley', 'langermann', 'rastrigin', 'trid',
                     'sphere', 'zakharov', 'dixon_price', 'styblinski_tang']

    np.random.seed(seed)
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
        for j in range(n_setups):
            np.random.seed(seed)
            torch.manual_seed(seed)
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

            set_random_seed(seed)
            task.restart()
            search_space = task.get_search_space()
            optimizer.restart()

            #initialize starting data
            x_init = search_space.sample(n_init)
            y_init = task(x_init)
            optimizer.initialize(x_init, y_init)

            for i in range(n_iterations):

                # get suggestion
                suggest_stopwatch.start()
                x = optimizer.suggest()
                suggest_stopwatch.stop()

                y = task(x)

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
                    y[0, 0],
                    float(optimizer.best_y)
                )

            setup_number += 1

    # clear some memory
    del optimizer, task, search_space

    return experiment_data


if __name__ == "__main__":
    # do argparse business
    parser = argparse.ArgumentParser(description='evaluates PFN performance')
    parser.add_argument('--task', '-t', type=str, help='which function to execute')
    parser.add_argument('--model_number', '-m', type=int, help='which model to evaluate')
    parser.add_argument('--prior', '-p', type=str, help="name of the pfn's prior")
    parser.add_argument('--seed', '-s', type=int, help='random seed')
    parser.add_argument('--acq_func', '-af', required=False, default='ei', type=str)
    parser.add_argument('--acq_opt', '-ao', required=False, default='is', type=str)
    args = parser.parse_args()

    task = args.task
    data_file_no = args.model_number
    prior = args.prior.lower()
    seed = args.seed

    prior_name_dict = {
        # maps lowercase to actual name used by mcbo
        'cocabo': 'CoCaBO',
        'casmo': 'Casmopolitan',
        'casmopolitan': 'Casmopolitan',
        'bodi': 'BODi',
        'novel': 'novel',
        'mixed': 'mixed'
    }

    if prior_name_dict[prior] == "CoCaBO":
        data_folder = ''
    elif prior_name_dict[prior] == 'Casmopolitan':
        data_folder = 'casmo_'
    elif prior_name_dict[prior] == 'BODi':
        data_folder = 'bodi_'
    elif prior_name_dict[prior] == 'novel':
        data_folder = 'novel_'
    elif prior_name_dict[prior] == 'mixed':
        data_folder = 'mixed_'
    else:
        raise ValueError('invalid prior')

    # suppress warnings
    warnings.filterwarnings('ignore')

    # set the device and other parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    directory = ''

    MODEL_FILENAME = directory+'PFNs4MVBO/trained_models/pfn_' + prior + '_' + str(data_file_no) + '.pth'

    # settings for BO runs
    n_init = 10
    n_iterations = 100
    n_setups = 10
    seed = 1235

    if task == 'regression':
        model_pfn_kwargs = {
            'pfn': MODEL_FILENAME,
            'num_out': 1,
            'dtype': torch.float32,
            'device': device
        }
        results_pfn = do_regression_evaluation(PFNModel, n_setups=100, n_test_points=1000, seed=seed,
                                               **model_pfn_kwargs)

        results_pfn.to_csv(directory+'PFNs4MVBO/'+data_folder+'evaluation_data/regression_results_'+str(data_file_no)+'.csv', index=False)

    elif task == 'overlap' or task == 'divergence':
        model_pfn_kwargs = {
            'pfn': MODEL_FILENAME,
            'num_out': 1,
            'dtype': torch.float32,
            'device': device
        }

        results_div = compare_pfn_vs_mcbo(PFNModel, n_setups=100, n_test_points=1000, seed=seed,
                                          mcbo_name=prior_name_dict[prior],
                                          use_saved_outputs=True,
                                          **model_pfn_kwargs)
        results_div.to_csv(directory+'PFNs4MVBO/'+data_folder+'evaluation_data/divergence_results_'+str(data_file_no)+'.csv', index = False)

    elif task == 'verify_divergence':
        model_pfn_kwargs = {
            'pfn': MODEL_FILENAME,
            'num_out': 1,
            'dtype': torch.float32,
            'device': device
        }

        results_div = compare_pfn_vs_mcbo(PFNModel, n_setups=100, n_test_points=1000, seed=seed, use_saved_outputs=True,
                                          **model_pfn_kwargs)
        results_div.to_csv(directory+'PFNs4MVBO/'+data_folder+'evaluation_data/divergence_verification_results_'+str(data_file_no)+'.csv', index = False)

    elif task == 'BO' or task == 'optimization':
        # first we do run using the pfn's acquisition function
        optimizer_kwargs = {
            'pfn_file': MODEL_FILENAME,
            'device': device,
            'fast': True
        }

        # add prior-dependent acq_func and acq_optim
        if prior.lower() == 'cocabo':
            optimizer_kwargs['acq_func'] = 'ei'
            optimizer_kwargs['acq_optim_name'] = 'mab'
        elif prior.lower() == 'casmo' or prior.lower() == 'casmopolitan':
            optimizer_kwargs['acq_func'] = 'ei'
            optimizer_kwargs['acq_optim_name'] = 'is'
            optimizer_kwargs['tr_id'] = 'basic'
        elif prior.lower() == 'bodi':
            optimizer_kwargs['acq_func'] = 'ei'
            optimizer_kwargs['acq_optim_name'] = 'is'
        elif prior.lower() == 'mixed':
            optimizer_kwargs['acq_func'] = args.acq_func
            optimizer_kwargs['acq_optim_name'] = args.acq_opt
        elif prior.lower() == 'novel':
            optimizer_kwargs['acq_func'] = args.acq_func
            optimizer_kwargs['acq_optim_name'] = args.acq_opt

        results_pfn_bo = do_validation_experiment(MVPFNOptimizer,
                                                  n_init=n_init,
                                                  n_iterations=n_iterations,
                                                  n_setups=n_setups,
                                                  seed=seed,
                                                  **optimizer_kwargs)

        torch.cuda.empty_cache()

        # now we use mcbo aq funcs that use mean and variance as inputs
        optimizer_kwargs = {
            'pfn_file': MODEL_FILENAME,
            'use_pfn_acq_func': False,
            'device': device,
            'fast': True
        }

        # add prior-dependent acq_func and acq_optim
        if prior.lower() == 'cocabo':
            optimizer_kwargs['acq_func'] = 'ei'
            optimizer_kwargs['acq_optim_name'] = 'mab'
        elif prior.lower() == 'casmo' or prior.lower() == 'casmopolitan':
            optimizer_kwargs['acq_func'] = 'ei'
            optimizer_kwargs['acq_optim_name'] = 'is'
            optimizer_kwargs['tr_id'] = 'basic'
        elif prior.lower() == 'bodi':
            optimizer_kwargs['acq_func'] = 'ei'
            optimizer_kwargs['acq_optim_name'] = 'is'
        elif prior.lower() == 'mixed':
            optimizer_kwargs['acq_func'] = args.acq_func
            optimizer_kwargs['acq_optim_name'] = args.acq_opt
        elif prior.lower() == 'novel':
            optimizer_kwargs['acq_func'] = args.acq_func
            optimizer_kwargs['acq_optim_name'] = args.acq_opt

        results_pfn_mcbo = do_validation_experiment(MVPFNOptimizer,
                                                    n_init=n_init,
                                                    n_iterations=n_iterations,
                                                    n_setups=n_setups,
                                                    seed=seed,
                                                    **optimizer_kwargs)

        torch.cuda.empty_cache()

        results_bo = pd.concat([results_pfn_bo,
                                results_pfn_mcbo], ignore_index=True)
        results_bo.to_csv(f'{directory}PFNs4MVBO/{data_folder}evaluation_data/BO_results_pfn_{str(data_file_no)}_{args.acq_func}_{args.acq_opt}.csv')

    elif task == 'baseline' or 'baselines':
        # Run MCBO Baselines and Random Baseline

        # CoCaBO regression
        model_kwargs = {'dtype': torch.float32, 'device': device}
        results_mcbo = do_regression_evaluation('CoCaBO', n_setups=100, n_test_points=1000, seed=seed, **model_kwargs)
        results_mcbo.to_csv(directory+'PFNs4MVBO/evaluation_data/regression_results_cocabo.csv')

        # Casmopolitan regression
        model_kwargs = {'dtype': torch.float32, 'device': device}
        results_mcbo = do_regression_evaluation('Casmopolitan', n_setups=100, n_test_points=1000, seed=seed, **model_kwargs)
        results_mcbo.to_csv(directory+'PFNs4MVBO/evaluation_data/regression_results_casmo.csv')

        # BODi regression
        model_kwargs = {'dtype': torch.float32, 'device': device}
        results_mcbo = do_regression_evaluation('BODi', n_setups=100, n_test_points=1000, seed=seed, **model_kwargs)
        results_mcbo.to_csv(directory + 'PFNs4MVBO/evaluation_data/regression_results_bodi.csv')

        # NOW WE DO BO
        optimizer_kwargs = {'n_init': 10,
                            'device': device}

        # For each mcbo alg, do BO runs
        for alg in ['CoCaBO', 'Casmopolitan', 'BODi']:
            results_mcbo_bo = do_validation_experiment(BO_ALGOS[alg].build_bo,
                                                        # n_init=n_init,
                                                        n_iterations=n_iterations,
                                                        n_setups=n_setups,
                                                        seed=seed,
                                                        **optimizer_kwargs)
            results_mcbo_bo.to_csv(directory+f'PFNs4MVBO/evaluation_data/BO_results_{alg}_baseline.csv')

        # Random Baseline
        optimizer_kwargs = {}
        results_random_bo = do_validation_experiment(RandomSearch,
                                                     n_init=n_init,
                                                     n_iterations=n_iterations,
                                                     n_setups=n_setups,
                                                     seed=seed,
                                                     **optimizer_kwargs)

        results_random_bo.to_csv(directory+'PFNs4MVBO/evaluation_data/BO_results_random_baseline.csv')

    else:
        raise ValueError('Invalid Task')