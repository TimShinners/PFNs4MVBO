import torch
import numpy as np
from pfns4bo import priors, encoders, utils, bar_distribution, train
from ConfigSpace import hyperparameters as CSH
from pfns4mvbo.priors import mixed_prior
from pfns4mvbo.get_categorical_encoder import get_categorical_encoder
from pfns4mvbo.utils import get_validation_data
import os
import argparse
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Trains a new PFN')
parser.add_argument('-lr', '--learning_rate', type=float)
parser.add_argument('-ce', '--categorical_encoding', type=str, required=False, default='default')
parser.add_argument('-w', '--weights', type=str, required=False, default='1,1,1')
args = parser.parse_args()

weights = args.weights.split(',')
assert len(weights) == 3
weights = np.array([int(w) for w in weights]).astype(float)
weights = weights / weights.sum()

config_mixed = {
     'priordataloader_class': priors.get_batch_to_dataloader(
         priors.get_batch_sequence(
             mixed_prior.get_batch,
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
            'prior_probabilities': weights,
            'categorical_encoding': 'default',
            'cocabo_hps': {
                'categorical_encoding': args.categorical_encoding,
                'sample_hyperparameters': True,
                'hp_sampling_method': 'authentic', # options are from_distributions, authentic, or from_dataset
                'n_fake_data': -1,
                'power_normalization': False,
                'unused_feature_likelihood': 0.3,
                'observation_noise': 0.7
            },
            'casmo_hps': {
                'categorical_encoding': 'default',
                'sample_hyperparameters': True,
                'hp_sampling_method': 'authentic',
                'n_fake_data': 20,
                'power_normalization': False,
                'unused_feature_likelihood': 0.3,
                'observation_noise': 0.9
            },
            'bodi_hps': {
                'categorical_encoding': 'default',
                'sample_hyperparameters': True,
                'hp_sampling_method': 'authentic',
                'n_fake_data': -1,
                'n_fake_data_method': 'cos',
                'sample_raw_hps': False,
                'power_normalization': False,
                'unused_feature_likelihood': 0.3,
                'observation_noise': True
            }
        }
      },
     'epochs': 200,
     'lr': args.learning_rate,
     'bptt': 60,
     'single_eval_pos_gen': utils.get_uniform_single_eval_pos_sampler(50, min_len=1), #<function utils.get_uniform_single_eval_pos_sampler.<locals>.<lambda>()>,
     'aggregate_k_gradients': 2,
     'nhid': 1024,
     'steps_per_epoch': 512,
     'weight_decay': 0.0,
     'train_mixed_precision': False,
     'efficient_eval_masking': True,
     'nlayers': 12}

# set up the categorical encoding
categorical_encoder = get_categorical_encoder(config_mixed['extra_prior_kwargs_dict']['hyperparameters']['categorical_encoding'])
config_mixed['extra_prior_kwargs_dict']['hyperparameters']['categorical_encoding_function'] = categorical_encoder


# now let's add the criterions, where we decide the border positions based on the prior
def get_ys(config, device='cuda:0'):
    bs = 128
    all_targets = []
    for num_hps in [2, 8, 12]: # a few different samples in case the number of features makes a difference in y dist
        b = config['priordataloader_class'].get_batch_method(bs, 1000, num_hps, epoch=0, device=device,
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
validation = get_validation_data(config_mixed, device=device)

integer = np.random.randint(1000000)
print('LOSS CURVE FILE NUMBER: ', integer, flush=True)
loss_curve_filename = 'PFNs4MVBO/loss_curves/loss_curve'+str(integer)+'.npy'
np.save(loss_curve_filename, np.zeros(config_mixed['epochs']))


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


ret = train.train(epoch_callback=epoch_callback, **add_criterion(config_mixed, device=device))

if ret is not None:
    model = ret[2]

    # can't save the categorical encoder with the model so we delete it
    del config_mixed['extra_prior_kwargs_dict']['hyperparameters']['categorical_encoding_function']

    # add model info
    model.info = {
        'prior': 'MIXED',
        'hyperparameters': config_mixed['extra_prior_kwargs_dict']['hyperparameters'],
        'learning_rate': config_mixed['lr'],
        'bptt': config_mixed['bptt'],
        'loss_curve': torch.tensor(np.load(loss_curve_filename))
    }

    # get model number
    model_number = 0
    filenames = os.listdir('PFNs4MVBO/trained_models')
    for filename in filenames:
        filename_prefix = 'pfn_mixed_'
        if filename[0:len(filename_prefix)] == filename_prefix:
            number = int(filename[len(filename_prefix):filename.index('.')])
            model_number = max(model_number, number+1)

    torch.save(model, f'PFNs4MVBO/trained_models/pfn_mixed_{model_number}.pth')
else:
    print("Nothing returned", ret)
