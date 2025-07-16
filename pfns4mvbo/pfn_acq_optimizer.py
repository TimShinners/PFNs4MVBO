import math

import torch
import numpy as np
import pandas as pd
import os
import pfns4bo
from pfns4bo.pfn_bo_bayesmark import PFNOptimizer
import mcbo
try:
    from .pfn_model import PFNModel
except:
    from pfn_model import PFNModel
from mcbo import task_factory
from mcbo.optimizers.bo_builder import BoBuilder
from mcbo.search_space import SearchSpace


class PFNAcqOptimizer():

    def __init__(self, pfn_model: PFNModel, search_space: SearchSpace, device, **kwargs):
        assert isinstance(search_space, SearchSpace), 'search_space needs to be an instance of mcbo.search_space.SearchSpace'
        assert isinstance(pfn_model, PFNModel)
        self.search_space = search_space
        self.device = device
        self.pfn_model = pfn_model

        self.optim_args = kwargs
        self.optim_args.setdefault('optimizer', 'adam')
        self.optim_args.setdefault('parallelize_restarts', True)

        self.input_constraints = self.get_input_constraints(self.search_space,
                                                            self.pfn_model.categorical_encoder)

    def get_input_constraints(self, search_space, categorical_encoder):
        # input_constraints are formatted as [lb, lb, lb, ...]
        #                                    [ub, ub, ub, ...]
        input_constraints = search_space.sample(2)

        for i, param in enumerate(search_space.param_names):
            if not (search_space.params[param].is_cont or search_space.params[param].is_nominal or search_space.params[param].is_disc):
                print(f"Invalid param type: {param}")
                print('cont', search_space.params[param].is_cont)
                print('nominal', search_space.params[param].is_nominal)
                print('disc', search_space.params[param].is_disc)
                print('ordinal', search_space.params[param].is_ordinal)
                print('permutation', search_space.params[param].is_permutation)

            if search_space.params[param].is_nominal:
                input_constraints[param][0] = min(search_space.params[param].categories)
                input_constraints[param][1] = max(search_space.params[param].categories)
            elif search_space.params[param].is_cont:
                input_constraints[param][0] = search_space.params[param].lb
                input_constraints[param][1] = search_space.params[param].ub
            elif search_space.params[param].is_disc:
                assert search_space.params[param].param_dict['type'] == 'int'
                input_constraints[param][0] = search_space.params[param].lb
                input_constraints[param][1] = search_space.params[param].ub

        # convert to tensor
        input_constraints = search_space.transform(input_constraints)

        # do categorical encoding
        assert self.pfn_model.categorical_encoder_name == 'default', "categorical encoding not implemented for discrete values"
        input_constraints = categorical_encoder(input_constraints, search_space)

        return input_constraints.to(device=self.device)

    def optimize(self):
        # produces one suggestion

        if self.optim_args['optimizer'].lower() == 'adam' and not self.optim_args['parallelize_restarts']:
            best_x = self.search_space.sample()
            best_x = self.search_space.transform(best_x).to(device=self.device)
            best_x = self.pfn_model.categorical_encoder(best_x, self.search_space)
            best_acq_value = -self.pfn_model.evaluate_acq_func(best_x, **{'best_f': self.pfn_model.fit_y.max()}).to(device=self.device)
            for restart in range(self.optim_args['n_restarts']):

                # get initial x
                x = self.search_space.sample(1)
                x = self.search_space.transform(x)
                x = self.pfn_model.categorical_encoder(x, self.search_space)
                x = x.clone().to(device=self.device).requires_grad_(True)

                optimizer = torch.optim.Adam([x], lr=0.01)



                # optimization loop
                for n in range(self.optim_args['n_iter']):
                    # attempting to use torch.clamp caused issues,
                    # instead I will simply break if we move out
                    # of bounds and rely on using a high number of
                    # restarts to make good optimizer performance
                    if not torch.all(x == torch.clamp(x, self.input_constraints[0], self.input_constraints[1])):
                        break

                    optimizer.zero_grad()
                    acq_value = -self.pfn_model.evaluate_acq_func(x, **{'best_f': self.pfn_model.fit_y.max()}).to(device=self.device)
                    acq_value.backward(retain_graph=True)
                    optimizer.step()

                    if acq_value < best_acq_value:
                        best_x = x.clone().to(device=self.device)
                        best_acq_value = acq_value.to(device=self.device)

        elif self.optim_args['optimizer'].lower() == 'adam' and self.optim_args['parallelize_restarts']:
            best_x = self.search_space.sample()
            best_x = self.search_space.transform(best_x).to(device=self.device)
            best_x = self.pfn_model.categorical_encoder(best_x, self.search_space)
            best_acq_value = -self.pfn_model.evaluate_acq_func(best_x, **{'best_f': self.pfn_model.fit_y.max()}).to(device=self.device)

            # get initial x
            x = self.search_space.sample(self.optim_args['n_restarts'])
            x = self.search_space.transform(x)
            x = self.pfn_model.categorical_encoder(x, self.search_space)
            x = x.clone().to(device=self.device).requires_grad_(True)

            optimizer = torch.optim.Adam([x], lr=0.01)

            # optimization loop
            for n in range(self.optim_args['n_iter']):

                optimizer.zero_grad()
                acq_values = -self.pfn_model.evaluate_acq_func(x, **{'best_f': self.pfn_model.fit_y.max()}).to(device=self.device)
                acq_value = acq_values.sum()
                acq_value.backward(retain_graph=True)
                optimizer.step()

                # check which x in bounds
                in_bounds = torch.all(x == torch.clamp(x, self.input_constraints[0], self.input_constraints[1]), axis=1)

                if in_bounds.to(dtype=int).sum() == 0:
                    # none of the x's are in bounds so we break
                    break

                best_acq_this_round = acq_values[in_bounds].min().to(device=self.device)
                if best_acq_this_round < best_acq_value:
                    best_x = x[acq_values == best_acq_this_round, :].clone().to(device=self.device)
                    best_acq_value = best_acq_this_round.to(device=self.device)
                    # print(f'iteration {n}, best x {best_x}, best y {best_acq_value}')

        else:
            raise

        # round nominals
        assert self.pfn_model.categorical_encoder_name == 'default', 'inverse categorical encodings not implemented'
        nominal_dims = torch.tensor(self.search_space.nominal_dims).to(device=self.device, dtype=torch.int).detach()

        best_x[:, nominal_dims] = torch.round(best_x[:, nominal_dims])

        best_x = best_x.detach()

        return best_x


























