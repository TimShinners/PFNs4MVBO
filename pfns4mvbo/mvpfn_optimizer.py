import torch
import numpy as np
import pandas as pd
import os
import pfns4bo
from mcbo.trust_region.casmo_tr_manager import CasmopolitanTrManager
from mcbo.trust_region.tr_manager_base import TrManagerBase

import mcbo
from mcbo import task_factory
from mcbo.optimizers.bo_builder import BoBuilder
from mcbo.optimizers.bo_base import BoBase
from mcbo.optimizers.optimizer_base import OptimizerBase
from mcbo.acq_funcs import acq_factory
import re
try:
    from .pfn_model import PFNModel
    from .pfn_acq_func import PFNAcqFunc
    from .pfn_acq_optimizer import PFNAcqOptimizer
except:
    from pfn_model import PFNModel
    from pfn_acq_func import PFNAcqFunc
    from pfn_acq_optimizer import PFNAcqOptimizer


class MVPFNOptimizer(OptimizerBase):
    # a class that combines the model, acquisition function,
    # and acq func optimizer to be a full BO algorithm
    def __init__(self, search_space, pfn, acq_func, acq_optim_name, acq_optim_kwargs={}, n_init=1, use_pfn_acq_func=True, device='cpu', dtype=torch.float64, input_constraints=None, tr_id=None, fast=True, **config):
        '''
        search_space: an instance of a search space from mcbo, gives info for features
        pfn_file: filename for the trained pfn to use as surrogate, should be "asdf.pth"
        acq_func: string saying which acquisition function to use, like 'ei'
        acq_func_optim: string, which acq func optimizer to use, like 'mab'
        categorical_encoding: string, specifies how to encode categories to numbers
        config: dict, has configuration arguments
            {
            'device': cpu or cuda, default cpu
            'configPFN': config for the PFN optimizer, scroll down to see default
            'acq_optim_name': name of the acquisition function optimizer, default 'mab'
            'input_space_constraints': default None
            'acq_optim_kwargs': default None
            }
        '''
        super().__init__(search_space=search_space,
                         input_constraints=None,
                         dtype=dtype)


        if pfn.endswith('.pth') or pfn.endswith('.pt'):
            # pfn is already a filepath 
            pfn_file = pfn
        else:
            import importlib.resources as pkg_resources

            if pfn.lower() == 'cocabo':
                pfn_name = 'pfn_cocabo_51'
            elif pfn.lower() in ['casmo', 'casmopolitan']:
                pfn_name = 'pfn_casmopolitan_16'
            elif pfn.lower() == 'bodi':
                pfn_name = 'pfn_bodi_24'
            else:
                raise ValueError("pfn must be a filepath to a pfn, or one of 'cocabo', 'casmo' or 'bodi'")

            with pkg_resources.path('pfns4mvbo', f'{pfn_name}.pth') as p:
                pfn_file = str(p)


        if len(re.findall(r'\d+', pfn_file)) > 0:
            self.name = 'MVPFN_'+re.findall(r'\d+', pfn_file)[0]+f',{"pfn" if use_pfn_acq_func else "mcbo"}AcqFunc'
        else:
            self.name = 'MVPFN' + f',{"pfn" if use_pfn_acq_func else "mcbo"}AcqFunc'

        self.device = device

        self.n_init = n_init

        self.search_space = search_space

        self.model_pfn = PFNModel(pfn_file, search_space, 1, torch.float32, device)

        try:
            categorical_encoding = self.pfn.info['training_hyperparameters']['categorical_encoding']

        except AttributeError:
            categorical_encoding = 'default'

        self.categorical_encoding = categorical_encoding

        if use_pfn_acq_func:
            # uses pfn's riemann distribution for acq
            self.acq_func = PFNAcqFunc(acq_func)
            self.model_pfn.set_acq_func(acq_func)
        else:
            # uses normal distributions
            self.acq_func = acq_factory(acq_func)

        self.acq_optim_name = acq_optim_name
        if acq_optim_name == 'pfn':
            # when forming a suggestion, we pretend the search_space
            # is fully continuous and optimize through that,
            # rounding the nominal inputs
            self.acq_optim_kwargs = acq_optim_kwargs
            self.acq_optimizer = PFNAcqOptimizer(self.model_pfn, self.search_space, self.device, **self.acq_optim_kwargs)
        else:
            # fix stupid bug
            try:
                if self.device == 'cpu' or self.device == torch.device('cpu'):
                    # fixing a weird bug... If you pass a device on cpu
                    # then I think it sets some global variable that
                    # messes other stuff up later on
                    1/0
                self.acq_optimizer = BoBuilder.get_acq_optim(search_space,
                                                             acq_optim_name,
                                                             input_constraints=input_constraints,
                                                             device=device)
            except:
                self.acq_optimizer = BoBuilder.get_acq_optim(search_space,
                                                             acq_optim_name,
                                                             input_constraints=input_constraints)

        if fast:
            # speed up acq optimization
            self.acq_optimizer.cont_n_iter = 10

            # for interleaved search
            self.acq_optimizer.n_restarts = 2
            self.acq_optimizer.nominal_tol = 100
            self.acq_optimizer.n_iter = 10

        self.tr_id = tr_id
        assert (tr_id == 'basic' and self.acq_optim_name != 'pfn') or tr_id is None
        self.tr_manager = BoBuilder.get_tr_manager(tr_id=self.tr_id, search_space=search_space, model=self.model_pfn, n_init=n_init,
                                                   )#**tr_kwargs)

    def name(self) -> str:
        name = ""
        name += self.model.name
        name += " - "
        if self.tr_manager is not None:
            name += f"Tr-based "
        name += f"{self.acq_optimizer.name} acq optim"
        return name

    def model_name(self) -> str:
        return self.model_pfn.name()

    def acq_func_name(self) -> str:
        return self.acq_func.name()

    def acq_opt_name(self) -> str:
        return self.acq_optimizer.name

    def initialize(self, x: pd.DataFrame, y: np.ndarray) -> None:
        assert y.ndim == 2
        assert x.ndim == 2
        assert y.shape[1] == 1
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == self.search_space.num_dims

        # transform data
        x_new = self.transform(x)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)

        # record new additions
        self.x = x_new
        self.y = y

        self.best_y = self.y.min()
        self._best_x = self.x[self.y.argmin(), :]

        # fit the model
        self.model_pfn.fit(self.x, self.y)

        if self.tr_manager is not None:
            self.tr_manager.append(x_new, y)

    def restart(self):
        self.x = None
        self.y = None
        self._best_x = None
        self.best_y = None
        self.model_pfn.restart()
        if self.tr_manager is not None:
            self.tr_manager.restart()

    def set_x_init(self, x: pd.DataFrame) -> None:
        self.x_init = x

    def tr_name(self) -> str:
        if self.tr_manager is None:
            return "no-tr"
        elif isinstance(self.tr_manager, CasmopolitanTrManager):
            return "basic"
        else:
            raise ValueError(self.tr_manager)

    def transform(self, X_pd):
        x = self.search_space.transform(X_pd)
        '''
        # we only transform within the model
        if self.categorical_encoding == 'default':
            return x
        else:
            raise NotImplementedError(f"categorical encoding {self.categorical_encoding} not implemented")
        '''
        return x

    def method_suggest(self, n_suggestions: int = 1) -> pd.DataFrame:
        assert n_suggestions == 1
        # produces one suggestion
        if (not hasattr(self, 'x')) or self.x is None:
            # if no observations, random suggest
            return self.search_space.sample()

        if self.acq_optim_name == 'pfn':
            # use optimizer from pfns4bo
            x_next = self.acq_optimizer.optimize()
        else:
            x_next = self.acq_optimizer.optimize(
                x=self._best_x,
                n_suggestions=1,
                x_observed=self.x,
                model=self.model_pfn,
                acq_func=self.acq_func,
                acq_evaluate_kwargs={'best_y': self.best_y},
                tr_manager=self.tr_manager,
                **{}
            )

        # print('x_next', x_next)
        # print('is nan', torch.any(torch.isnan(x_next)))
        if torch.any(torch.isnan(x_next)):
            print('optimizer returned NaN values, suggesting random point')
            return self.search_space.sample()

        x_next = self.search_space.inverse_transform(x_next)

        return x_next

    def method_observe(self, x, y):

        # transform data
        x_new = self.transform(x)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)

        # record new additions
        if hasattr(self, 'x') and hasattr(self, 'y') and isinstance(self.x, torch.Tensor):
            self.x = torch.vstack([self.x, x_new])
            self.y = torch.vstack([self.y, y])
        else:
            # initialize x and y
            self.x = x_new
            self.y = y

        if self.tr_manager is not None:
            if len(self.tr_manager.data_buffer) > self.n_init:
                self.tr_manager.adjust_tr_radii(y)
            self.tr_manager.append(x_new, y)
            self.tr_manager.adjust_tr_center()

        # update best x and y
        self.best_y = self.y.min()
        self._best_x = self.x[self.y.argmin(), :]

        # fit the model
        self.model_pfn.fit(self.x, self.y)


if __name__ == "__main__":
    task_kws = dict(variable_type=['num', 'nominal'],
                    num_dims=[1, 1],
                    lb=-1,
                    ub=1,
                    num_categories=[10, 2])

    task = task_factory(task_name="ackley", **task_kws)

    search_space = task.get_search_space()

    mvpfn = MVPFNOptimizer(search_space=search_space, pfn_file='../trained_models/test_model_00.pth', acq_func='pi',
                           acq_optim_name='mab', use_pfn_acq_func=False, device='cpu')
    print('here')

    from mcbo.utils.experiment_utils import run_experiment
    run_experiment(
        task,
        optimizers=[mvpfn],
        random_seeds=[555],
        max_num_iter=5,
        save_results_every=100,
        very_verbose=False,
        )
    print('here2')

    X_pd = search_space.sample(3)
    y = task(X_pd)
    mvpfn.observe(X_pd, y)
    best_y = y.min()
    print(best_y)

    for i in range(30):
        new_x = mvpfn.suggest()
        new_y = task(new_x)
        mvpfn.observe(new_x, new_y)

        if new_y < best_y:
            best_y = new_y
        print(new_y, best_y)



































