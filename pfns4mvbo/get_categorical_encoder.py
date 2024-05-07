import torch
from mcbo.search_space import SearchSpace

# this file defines a function that takes a name,
# and outputs a function that performs categorical
# encodings given x and an input space

def get_categorical_encoder(name: str):
    if name == 'default':
        def categorical_encoder(x: torch.Tensor, input_space: SearchSpace):
            return x

    elif name == 'add_2':
        # removes any ambiguity between numerical and nominal variables
        def categorical_encoder(x: torch.Tensor, input_space: SearchSpace):
            nominal_dims = torch.tensor(input_space.nominal_dims).to(device=x.device, dtype=torch.int).detach()
            # transform nominal dims
            x[:, nominal_dims] = x[:, nominal_dims] + 2
            return x

    elif name == 'scale':
        # puts numerical and nominal variables on the same [0,1] scale
        def categorical_encoder(x: torch.Tensor, input_space: SearchSpace):
            # take categorical variables, rescale to [0,1]
            nominal_dims = torch.tensor(input_space.nominal_dims).to(device=x.device, dtype=torch.int).detach()
            nominal_lb = torch.tensor(input_space.nominal_lb).to(device=x.device, dtype=x.dtype).detach()
            nominal_ub = torch.tensor(input_space.nominal_ub).to(device=x.device, dtype=x.dtype).detach()
            # transform nominal dims
            x[:, nominal_dims] = ((x[:, nominal_dims] - nominal_lb) / (nominal_ub - nominal_lb))
            return x

    elif name == 'scale_add_2':
        def categorical_encoder(x: torch.Tensor, input_space: SearchSpace):
            # take categorical variables, rescale to [0,1], then add 2
            nominal_dims = torch.tensor(input_space.nominal_dims).to(device=x.device, dtype=torch.int).detach()
            nominal_lb = torch.tensor(input_space.nominal_lb).to(device=x.device, dtype=x.dtype).detach()
            nominal_ub = torch.tensor(input_space.nominal_ub).to(device=x.device, dtype=x.dtype).detach()

            # transform nominal dims
            x[:, nominal_dims] = ((x[:, nominal_dims] - nominal_lb) / (nominal_ub - nominal_lb)) + 2

            return x

    else:
        raise ValueError('Categorical Encoding '+name+' is not defined')

    return categorical_encoder