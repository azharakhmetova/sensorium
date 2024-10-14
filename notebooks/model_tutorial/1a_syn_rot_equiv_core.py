import torch
device = torch.cuda.set_device('cuda:3')

import numpy as np
import pandas as pd

#import matplotlib.pyplot as plt
#import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from nnfabrik.builder import get_data, get_model, get_trainer

# loading the SENSORIUM dataset
filenames = ['../data/1_25_13_12_3_2', ]#['../data/static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip', ]

dataset_fn = 'sensorium.datasets.static_loaders'
dataset_config = {'paths': filenames,
                 'normalize': True,
                 'include_behavior': False,
                 'include_eye_position': False,
                 'batch_size': 128,
                 'scale':0.25,
                 }

dataloaders = get_data(dataset_fn, dataset_config)

model_fn = 'sensorium.models.ecker_core_full_gauss_readout'#'sensorium.models.stacked_core_full_gauss_readout'
model_config = {'pad_input': False,
  'stack': -1,
  'layers': 4,
  'input_kern': 9,
  'gamma_input': 6.3831,
  'gamma_readout': 0.0076,
  'hidden_kern': 7,
  'hidden_channels': 128,
  'num_rotations': 8, 
  'depth_separable': True,
  'grid_mean_predictor': {'type': 'cortex',
   'input_dimensions': 2,
   'hidden_layers': 1,
   'hidden_features': 30,
   'final_tanh': True},
  'init_sigma': 0.1,
  'init_mu_range': 0.3,
  'gauss_type': 'full',
  'shifter': False,
  # 'core': 'RotEquiv2dCore',
}

model = get_model(model_fn=model_fn,
                  model_config=model_config,
                  dataloaders=dataloaders,
                  seed=42,)

model

trainer_fn = "sensorium.training.standard_trainer"

trainer_config = {'max_iter': 200,
                 'verbose': False,
                 'lr_decay_steps': 4,
                 'avg_loss': False,
                 'lr_init': 0.009,
                 }

trainer = get_trainer(trainer_fn=trainer_fn, 
                     trainer_config=trainer_config)

validation_score, trainer_output, state_dict = trainer(model, dataloaders, seed=42)

torch.save(model.state_dict(), './model_checkpoints/sensorium_sota_model_2.pth')

# model.load_state_dict(torch.load("./model_checkpoints/pretrained/sensorium_sota_model_2.pth"));

# # %% [markdown]
# # ---

# # %% [markdown]
# # # Train a simple LN model

# # %% [markdown]
# # Our LN model has the same architecture as our CNN model (a convolutional core followed by a gaussian readout)
# # but with all non-linearities removed except the final ELU+1 nonlinearity.
# # Thus turning the CNN model effectively into a fully linear model followed by a single output non-linearity.
# # 

# # %%
# model_fn = 'sensorium.models.stacked_core_full_gauss_readout'
# model_config = {'pad_input': False,
#               'stack': -1,
#               'layers': 3,
#               'input_kern': 9,
#               'gamma_input': 6.3831,
#               'gamma_readout': 0.0076,
#               'hidden_kern': 7,
#               'hidden_channels': 64,
#               'grid_mean_predictor': {'type': 'cortex',
#               'input_dimensions': 2,
#               'hidden_layers': 1,
#               'hidden_features': 30,
#               'final_tanh': True},
#               'depth_separable': True,
#               'init_sigma': 0.1,
#               'init_mu_range': 0.3,
#               'gauss_type': 'full',
#               'linear': True
#                }
# model = get_model(model_fn=model_fn,
#                   model_config=model_config,
#                   dataloaders=dataloaders,
#                   seed=42,)

# # %%
# validation_score, trainer_output, state_dict = trainer(model, dataloaders, seed=42)

# # %%
# torch.save(model.state_dict(), './model_checkpoints/sensorium_ln_model.pth')

# # %%
# model.load_state_dict(torch.load("./model_checkpoints/pretrained/sensorium_ln_model.pth"));

# # %% [markdown]
# # ---


