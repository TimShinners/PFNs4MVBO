import torch
try:
    from .pfn_model import PFNModel
except:
    from pfn_model import PFNModel
from mcbo.acq_funcs.acq_base import AcqBase



class PFNAcqFunc():

    def __init__(self, acq_func, **kwargs):
        '''
        acq_func is a string saying which acquisition function to use
        for instance: 'ei'
        '''
        self.acq_func_name = acq_func
        self.kwargs = kwargs

    def __call__(self,
                 x: torch.Tensor,
                 model: PFNModel,
                 **kwargs
                 ) -> torch.Tensor:
        acq = self.evaluate(x, model, **kwargs)
        return acq

    def name(self):
        return self.acq_func_name

    def evaluate(self,
                 x: torch.Tensor,
                 model: PFNModel,
                 best_y=None,
                 **kwargs
                 ) -> torch.Tensor:
        '''
        assume best_y is not normalized!
        '''

        assert isinstance(model, PFNModel), 'Model must be PFNModel'

        if best_y is None:
            # try to get best_y from the model
            # REMINDER: we convert to fit_y in model.evaluate_acq_func()
            best_y = model._y.min()

        self.kwargs.update(kwargs)
        self.kwargs.update({'best_y': best_y,
                            'best_f': best_y})

        if not hasattr(model, 'acq_func_name') or model.acq_func_name != self.acq_func_name:
            # set acquisition function
            model.set_acq_func(self.acq_func_name)

        # I think we want to minimize, so we should use -acq!!
        acq = model.evaluate_acq_func(x, **self.kwargs)
        return -acq

