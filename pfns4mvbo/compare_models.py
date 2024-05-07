import torch
import numpy as np
import pandas as pd
import scipy
from mcbo import task_factory
from mcbo.search_space import SearchSpace
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
import pfns4bo
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from hebo.models.model_factory import get_model

# We define functions that run experiments and record data from different optimization runs

# first we define a helper function that will compare the
# pfn's output to the cocabo output


def normal_logits_overlap(mean_normal, variance_cocabo, logits, model_pfn):
    # this function calculates the overlap between the riemann (pfn) and normal
    # distributions (0 is bad, 1 is perfect overlap)
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
    std_cocabo = torch.sqrt(variance_cocabo)
    ub = torch.max((mean_normal + 4 * std_cocabo).max(), bucket_borders_pfn.max() + 0.1).item()
    lb = torch.min((mean_normal - 4 * std_cocabo).min(), bucket_borders_pfn.min() - 0.1).item()
    density = torch.max(torch.tensor(10000), (2 * (ub - lb) / bucket_widths_pfn).max()).item()
    small_buckets = np.linspace(lb, ub, int(density))
    small_bucket_width = small_buckets[1] - small_buckets[0]

    # reformat the probs and get pfn_pdf
    probs = torch.hstack([torch.zeros([probs.shape[0], 1]), probs, torch.zeros([probs.shape[0], 1])])
    pfn_indices = np.digitize(small_buckets, bucket_borders_pfn.detach())
    pfn_pdf = probs[:, pfn_indices].detach().numpy()
    pfn_pdf = pfn_pdf / np.hstack([1, bucket_widths_pfn.numpy(), 1])[pfn_indices]

    # calculate cocabo_pdf for each bucket
    normal_pdf = scipy.stats.norm.pdf(small_buckets, loc=mean_normal.detach().cpu(), scale=np.sqrt(variance_cocabo.detach().cpu()))

    # calculate the overlap as integrate min(pfn_pdf, cocabo_pdf)
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


def compare_models(model_1_name, model1_fit_and_predict, model1_kwargs,
                   model_2_name, model2_fit_and_predict, model2_kwargs,
                   n_setups, n_test_points, seed):
    # first let's do the prior it was trained on
    # then we will do on functions
    device = model1_kwargs.get('device', 'cpu')

    np.random.seed(seed)

    if model_1_name == 'HEBO' or model_2_name == 'HEBO':
        num_only = True
    else:
        num_only = False

    if num_only:
        numerical_dims = np.random.randint(2, 19, n_setups).astype(int)
    else:
        n_dims = np.random.randint(2, 19, n_setups).astype(int)
        numerical_dims = np.random.randint(1, n_dims).astype(int)
        categorical_dims = n_dims - numerical_dims

    n_training_points = np.arange(2, 11)**2

    comparison_data = pd.DataFrame(columns=[
        'trial_number',
        'setup_number',
        'model_name_1',
        'model_name_2',
        'task_name',
        'num_dims',
        'cat_dims',
        'seed',
        'n_training_data',
        'overlap'
    ])

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

                if num_only:
                    task_kws = dict(variable_type=['num'],
                                    num_dims=[int(numerical_dims[j])])
                else:
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

                output_1 = model1_fit_and_predict(x_train, y_train, x_test, search_space, **model1_kwargs)
                output_2 = model2_fit_and_predict(x_train, y_train, x_test, search_space, **model2_kwargs)

                # calculate the overlap
                if output_1[0] == 'normal' and output_2[0] == 'riemann':
                    overlap = normal_logits_overlap(output_1[1], output_1[2], output_2[1], output_2[2]).mean()
                if output_1[0] == 'riemann' and output_2[0] == 'normal':
                    overlap = normal_logits_overlap(output_2[1], output_2[2], output_1[1], output_1[2]).mean()
                if output_1[0] == 'normal' and output_2[0] == 'normal':
                    overlap = normal_normal_overlap(output_1[1], output_1[2], output_2[1], output_2[2]).mean()
                if output_1[0] == 'riemann' and output_2[0] == 'riemann':
                    raise NotImplementedError("PFN vs PFN comparison not yet implemented")

                # record data
                comparison_data.loc[len(comparison_data)] = (
                    trial_number,
                    setup_number,
                    model_1_name,
                    model_2_name,
                    task_name,
                    numerical_dims[j],
                    0 if num_only else categorical_dims[j],
                    seed,
                    n_dat,
                    overlap
                )
                trial_number += 1
            setup_number += 1

    return comparison_data


def HEBO_fit_and_predict(x_train: torch.Tensor,
                         y_train: torch.Tensor,
                         x_test: torch.Tensor,
                         search_space: SearchSpace,
                         **model_kwargs):
    variables = []
    for i, col in enumerate(search_space.df_col_names):
        variables += [
            {'name': col, 'type': 'num', 'lb': search_space.cont_lb[i], 'ub': search_space.cont_ub[i]}]

    space = DesignSpace().parse(variables)
    opt = HEBO(space)
    model_hebo = get_model(opt.model_name, opt.space.num_numeric, opt.space.num_categorical, 1,
                           **opt.model_config)

    x_train_pd = search_space.inverse_transform(x_train)
    X, Xe = space.transform(x_train_pd)
    model_hebo.fit(X.to(device='cpu', dtype=torch.float64), Xe.to(device='cpu', dtype=torch.float64),
                   y_train.to(device='cpu', dtype=torch.float64))

    test_x_pd = search_space.inverse_transform(x_test)
    X, Xe = space.transform(test_x_pd)
    mu_hebo, var_hebo = model_hebo.predict(X.to(device='cpu', dtype=torch.float64),
                                           Xe.to(device='cpu', dtype=torch.float64))

    del model_hebo

    return 'normal', mu_hebo, var_hebo


def PFN_fit_and_predict(x_train: torch.Tensor,
                        y_train: torch.Tensor,
                        x_test: torch.Tensor,
                        search_space: SearchSpace,
                        **model_kwargs):

    model_pfn = PFNModel(search_space=search_space, **model_kwargs)
    model_pfn.fit(x_train, y_train)

    logits = model_pfn.pfn(x_train.to(dtype=model_pfn.dtype, device=model_pfn.device),
                           model_pfn.y_to_fit_y(y_train).to(dtype=model_pfn.dtype, device=model_pfn.device),
                           x_test.to(dtype=model_pfn.dtype, device=model_pfn.device))

    return 'riemann', logits, model_pfn


def MCBO_fit_and_predict(x_train: torch.Tensor,
                         y_train: torch.Tensor,
                         x_test: torch.Tensor,
                         search_space: SearchSpace,
                         **model_kwargs):
    model = BO_ALGOS[model_kwargs['name']].build_bo(search_space=search_space, n_init=1).model

    model.fit(x_train, y_train)
    mu_mcbo, var_mcbo = model.predict(x_test)

    return 'normal', mu_mcbo, var_mcbo


def get_fit_and_predict_function(model_name):
    global device
    if model_name.lower() == 'hebo':
        model_kwargs = {'name': 'HEBO'}
        return HEBO_fit_and_predict, model_kwargs

    elif model_name == 'CoCaBO' or model_name == 'Casmopolitan':
        model_kwargs = {
            'name': model_name,
            'dtype': torch.float32,
            'device': device
        }
        return MCBO_fit_and_predict, model_kwargs

    elif 'pfn' in model_name:
        model_kwargs = {
            'num_out': 1,
            'dtype': torch.float32,
            'device': device
        }

        # define which pfn to load
        if 'hebo' in model_name:
            model_kwargs['pfn'] = pfns4bo.hebo_plus_model
        elif 'PFNs4MVBO/trained_models' in model_name:
            model_kwargs['pfn'] = model_name
        else:
            model_kwargs['pfn'] = 'PFNs4MVBO/trained_models/' + model_name

        return PFN_fit_and_predict, model_kwargs

    else:
        raise ValueError("Invalid model_name")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='compares posterior distributions between models')
    parser.add_argument('--model_1', '-m1', type=str, help='which model to compare')
    parser.add_argument('--model_2', '-m2', type=str, help='which model to compare')
    args = parser.parse_args()

    # set the device and other parameters
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    seed = 1235

    model_1_fit_and_predict, model_1_kwargs = get_fit_and_predict_function(args.model_1)
    model_2_fit_and_predict, model_2_kwargs = get_fit_and_predict_function(args.model_2)

    comparison = compare_models(args.model_1, model_1_fit_and_predict, model_1_kwargs,
                                args.model_2, model_2_fit_and_predict, model_2_kwargs,
                                n_setups=100, n_test_points=1000, seed=seed)
    comparison.to_csv(f'PFNs4MVBO/evaluation_data/{args.model_1}_vs_{args.model_2}_comparison.csv', index=False)
