import numpy as np
import torch
from pfns4bo import priors, encoders, utils, bar_distribution, train
from ConfigSpace import hyperparameters as CSH
from pfns4mvbo.priors import cocabo_prior
from pfns4mvbo.get_categorical_encoder import get_categorical_encoder
from pfns4mvbo.utils import get_validation_data
import os
import argparse
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Trains a new PFN')
parser.add_argument('-n', '--n_fake_data', type=int)
parser.add_argument('-lr', '--learning_rate', type=float)
parser.add_argument('-ce', '--categorical_encoding', type=str, required=False, default='default')
parser.add_argument('-fd', '--n_fake_data_method', type=str, required=False, default='normal')
args = parser.parse_args()

config_cocabo = {
     'priordataloader_class': priors.get_batch_to_dataloader(
         priors.get_batch_sequence(
             cocabo_prior.get_batch,
             priors.utils.sample_num_feaetures_get_batch,
         )
     ),
     'encoder_generator': encoders.get_normalized_uniform_encoder(encoders.get_variable_num_features_encoder(encoders.Linear)),
     'emsize': 512,
     'nhead': 4,
     'warmup_epochs': 5,
     'y_encoder_generator': encoders.Linear,
     'batch_size': 128,
     'scheduler': utils.get_cosine_schedule_with_warmup,
     'extra_prior_kwargs_dict': {'num_features': 18,
      'hyperparameters': {
            'categorical_encoding': args.categorical_encoding,
            'sample_hyperparameters': True,
            'hp_sampling_method': 'authentic', # options are from_distributions, authentic, or from_dataset
            'n_fake_data': args.n_fake_data,
            'n_fake_data_method': args.n_fake_data_method,
            'lamda_shape': 8,
            'outputscale_concentration': 10, # 0.8452312502679863,
            'outputscale_rate': 8, # 0.3993553245745406,
            'num_lengthscale_concentration': 2, # 1.2106559584074301,
            'num_lengthscale_rate': 2, # 1.5212245992840594,
            'cat_lengthscale_concentration': 2, # 1.2106559584074301,
            'cat_lengthscale_rate': 2, # 1.5212245992840594,
            'power_normalization': False,
            'unused_feature_likelihood': 0.3,
            'observation_noise': True}},
     'epochs': 200, # 100,
     'lr': args.learning_rate,#0.0001,
     'bptt': 60,
     'single_eval_pos_gen': utils.get_uniform_single_eval_pos_sampler(50, min_len=1), #<function utils.get_uniform_single_eval_pos_sampler.<locals>.<lambda>()>,
     'aggregate_k_gradients': 2,
     'nhid': 1024,
     'steps_per_epoch': 512, # 1024,
     'weight_decay': 0.0,
     'train_mixed_precision': False,
     'efficient_eval_masking': True,
     'nlayers': 12}

# set up the categorical encoding
categorical_encoder = get_categorical_encoder(config_cocabo['extra_prior_kwargs_dict']['hyperparameters']['categorical_encoding'])
config_cocabo['extra_prior_kwargs_dict']['hyperparameters']['categorical_encoding_function'] = categorical_encoder


# now let's add the criterions, where we decide the border positions based on the prior
def get_ys(config, device='cuda:0'):
    bs = 128
    all_targets = []
    for num_hps in [2,8,12]: # a few different samples in case the number of features makes a difference in y dist
        b = config['priordataloader_class'].get_batch_method(bs,1000,num_hps,epoch=0,device=device,
                                                              hyperparameters=
                                                              {**config['extra_prior_kwargs_dict']['hyperparameters'],
                                                               'num_hyperparameter_samples_per_batch': -1,})
        all_targets.append(b.target_y.flatten())
    return torch.cat(all_targets,0)


def add_criterion(config, device='cuda:0'):
    return {**config, 'criterion': bar_distribution.FullSupportBarDistribution(
        bar_distribution.get_bucket_limits(1000,ys=get_ys(config,device).cpu())
    )}


global validation
validation = get_validation_data(config_cocabo, device=device)

integer = np.random.randint(100000)
print('LOSS CURVE FILE NUMBER: ', integer, flush=True)
loss_curve_filename = 'PFNs4MVBO/loss_curves/loss_curve'+str(integer)+'.npy'
np.save(loss_curve_filename, np.zeros(config_cocabo['epochs']))


def epoch_callback(model, epoch, data_loader, scheduler):
    # function used for recording the loss at the end of each epoch
    # calculate loss here
    global validation
    with torch.no_grad():
        loss = torch.zeros(len(validation['train']))
        for i, train_data in enumerate(validation['train']):
            logits = model(train_data[0], train_data[1], validation['test'][i][0])
            targets = validation['test'][i][1]
            loss[i] = model.module.criterion.forward(logits, targets).mean()
        loss = loss.mean().item()
    loss_curve = np.load(loss_curve_filename)
    loss_curve[epoch-1] = loss
    np.save(loss_curve_filename, loss_curve)
    # can calculate overlap too later on
    return


ret = train.train(epoch_callback=epoch_callback, **add_criterion(config_cocabo,device=device))

if ret is not None:
    model = ret[2]

    # can't save the categorical encoder with the model so we delete it
    del config_cocabo['extra_prior_kwargs_dict']['hyperparameters']['categorical_encoding_function']

    # add model info
    model.info = {
        'prior': 'CoCaBO',
        'categorical_encoding': config_cocabo['extra_prior_kwargs_dict']['hyperparameters']['categorical_encoding'],
        'sample_hyperparameters': config_cocabo['extra_prior_kwargs_dict']['hyperparameters']['sample_hyperparameters'],
        'observation_noise': config_cocabo['extra_prior_kwargs_dict']['hyperparameters']['observation_noise'],
        'training_hyperparameters': config_cocabo['extra_prior_kwargs_dict']['hyperparameters'],
        'learning_rate': config_cocabo['lr'],
        'bptt': config_cocabo['bptt'],
        'loss_curve': torch.tensor(np.load(loss_curve_filename))
    }

    # get model number
    model_number = 0
    filenames = os.listdir('PFNs4MVBO/trained_models')
    for filename in filenames:
        if filename[0:11] == 'pfn_cocabo_':
            number = int(filename[11:filename.index('.')])
            model_number = max(model_number, number+1)

    torch.save(model, f'PFNs4MVBO/trained_models/pfn_cocabo_{model_number}.pth')
else:
    print("Nothing returned", ret)
