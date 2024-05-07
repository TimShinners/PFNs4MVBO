import torch
from mcbo.models.model_base import ModelBase
from mcbo.search_space import SearchSpace
from typing import Optional, List
try:
    from .get_categorical_encoder import get_categorical_encoder
except:
    from get_categorical_encoder import get_categorical_encoder


class PFNModel(ModelBase):
    def __init__(self, pfn, search_space: SearchSpace, num_out: int, dtype: torch.dtype, device: torch.device, **kwargs):
        super().__init__(search_space=search_space, num_out=num_out, device=device, dtype=dtype, **kwargs)

        if isinstance(pfn, str):
            pfn = torch.load(pfn)

        assert isinstance(pfn, torch.nn.Module), "pfn needs to be filepath (string) or the loaded torch module"
        self.pfn = pfn.to(dtype=self.dtype, device=self.device)

        # disable grad
        for param in self.pfn.parameters():
            param.requires_grad = False

        self.pfn.requires_grad_(requires_grad=False)

        try:
            self.categorical_encoder_name = self.pfn.info['training_hyperparameters']['categorical_encoding']

        except AttributeError:
            # in case PFN has no .info This
            # ensures compatibility with pfns4bo models
            self.categorical_encoder_name = 'default'
        except KeyError:
            # mixed pfns will throw a key error
            self.categorical_encoder_name = self.pfn.info['hyperparameters']['categorical_encoding']

        self.categorical_encoder = get_categorical_encoder(self.categorical_encoder_name)

    def fit(self, x: torch.Tensor, y: torch.Tensor, **kwargs):
        assert x.ndim == 2
        assert y.ndim == 2, y.shape
        assert x.shape[0] == y.shape[0]
        assert y.shape[1] == self.num_out, (y.shape, self.num_out)

        '''
        # Do we want to remove repeating samples? 
        # seems like repeated experiments with repeated 
        # results should decrease uncertainty?
        # Remove repeating data points
        x, y = remove_repeating_samples(x, y)
        '''

        self.x = x.to(dtype=self.dtype, device=self.device)
        self.x = self.categorical_encoder(self.x, self.search_space)
        self._y = y.to(dtype=self.dtype, device=self.device)
        self.fit_y = self.y_to_fit_y(y=self._y).to(dtype=self.dtype, device=self.device)
        return

    def fit_y_to_y(self, fit_y: torch.Tensor) -> torch.Tensor:
        return fit_y * self.y_std.to(fit_y) + self.y_mean.to(fit_y)

    def y_to_fit_y(self, y: torch.Tensor) -> torch.Tensor:
        # Normalise target values
        fit_y = (y - self.y_mean.to(y)) / self.y_std.to(y)
        return fit_y

    def name(self):
        return "PFN"

    def noise(self) -> torch.Tensor:
        assert 1 == 0, "noise not relevant for PFN"

    def restart(self):
        self.x = None
        self._y = None
        self.fit_y = None

    @torch.no_grad()
    def predict(self, x: torch.Tensor, **kwargs) -> (torch.Tensor, torch.Tensor):
        test_x = x.to(dtype=self.dtype, device=self.device)
        test_x = self.categorical_encoder(test_x, self.search_space)
        logits = self.pfn(self.x, self.fit_y, test_x)

        mu = self.pfn.criterion.mean(logits)
        var = self.pfn.criterion.variance(logits)

        mu = self.fit_y_to_y(mu)
        var = (var * self.y_std.to(mu) ** 2)

        return mu.unsqueeze(1), var.clamp(min=torch.finfo(var.dtype).eps).unsqueeze(1)

    @torch.no_grad()
    def predict_logits(self, x: torch.Tensor, **kwargs) -> (torch.Tensor, torch.Tensor):
        test_x = x.to(dtype=self.dtype, device=self.device)
        test_x = self.categorical_encoder(test_x, self.search_space)
        logits = self.pfn(self.x, self.fit_y, test_x)

        return logits

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        self.device = device
        self.dtype = dtype
        self.pfn.to(device=device, dtype=dtype)
        return super().to(device=device, dtype=dtype)

    def set_acq_func(self, name):

        self.acq_func_name = name

        if self.acq_func_name == 'ei':
            self.acq_func = self.pfn.criterion.ei
            self.acq_func_args = ['best_f']
        elif self.acq_func_name == 'pi':
            self.acq_func = self.pfn.criterion.pi
            self.acq_func_args = ['best_f']
        elif self.acq_func_name == 'ucb':
            self.acq_func = self.pfn.criterion.ucb
            self.acq_func_args = ['best_f', 'rest_prob']
        elif self.acq_func_name == 'icdf':
            self.acq_func = self.pfn.criterion.icdf
            self.acq_func_args = ['left_prob']
        elif self.acq_func_name == 'quantile':
            self.acq_func = self.pfn.criterion.quantile
            self.acq_func_args = ['center_prob']
        elif self.acq_func_name == 'mean':
            self.acq_func = self.pfn.criterion.mean
            self.acq_func_args = []
        elif self.acq_func_name == 'median':
            self.acq_func = self.pfn.criterion.median
            self.acq_func_args = []
        elif self.acq_func_name == 'mode':
            self.acq_func = self.pfn.criterion.mode
            self.acq_func_args = []
        else:
            raise NotImplementedError("This function is not implemented yet")

    def evaluate_acq_func(self, x, **kwargs):
        '''
        assume best_y is not normalized!
        '''
        if 'best_y' in kwargs or 'best_f' in kwargs:
            # normalize the best_y
            best_y = kwargs.get('best_y', kwargs['best_f'])
            best_y = -self.y_to_fit_y(best_y)
            kwargs.update({'best_y': best_y,
                           'best_f': best_y.item()})

        # check x dims
        if len(x.shape) < 2:
            x = x.unsqueeze(0)

        # get pfn output
        # pfn thinks we maximize the objective, but really we want to minimize,
        # so we negate self.fit_y
        logits = self.pfn(self.x, -self.fit_y, x.to(dtype=self.dtype, device=self.device))

        if torch.any(torch.isnan(logits)):
            print('NANS IN LOGITS')
            print('X SHAPE', self.x.shape)
            print('X VALUES')
            print(self.x)
            print('Y SHAPE', self.fit_y.shape)
            print('Y VALUES')
            print(self.fit_y)
            raise ValueError('pfn produced nan values')

        # filter unused arguments out of kwargs
        acq_kwargs = {key: value for key, value in kwargs.items() if key in self.acq_func_args}

        # evaluate acq func and return
        acq = self.acq_func(logits, **acq_kwargs)

        # REMINDER: we say -acq in PFNAcqFunc.evaluate()
        return acq

