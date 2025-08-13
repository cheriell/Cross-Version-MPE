import os
import torch
from torchinfo import summary
import os, sys
sys.path.insert(0, os.path.join(sys.path[0], '../..'))
import pandas as pd
import numpy as np

from experiments_icassp25.main.utils import (
    BaseLightningModule,
    path_from_config,
    update_log_file,
    get_criterion_fn,
)
from experiments_icassp25.models.aoe_cnns import aoe_cnn_model_no_recon
from experiments_icassp25.eval_metrics import calculate_eval_measures


#######################################################
# PyTorch Lightning Module for Teacher-Student training
#######################################################
    
class aoets_cnn_lightning_module(BaseLightningModule):

    def __init__(self, exp_configs, args):
        super().__init__(exp_configs, args)

        ae_layers = exp_configs['model_params']['ae_layers']
        self.model = aoe_cnn_model_no_recon(ae_layers)
        # Load the pretrained model
        if args.resume_training:
            path_model = path_from_config(exp_configs['path_configs']['path_model'], args)
            if os.path.exists(path_model):
                self.model.load_state_dict(torch.load(path_model))
            else:
                raise ValueError("The model file does not exist: {}".format(path_model))
        self.exp_configs = exp_configs
        self.args = args

        # Update the Log file
        update_log_file(self.path_log, "Model: {}".format(self.model.__class__.__name__))
        update_log_file(self.path_log, "\n{}".format(str(summary(self.model, input_size=(1, 6, 174, 216)))))

        # Loss functions
        self.criterion_fn = get_criterion_fn(self.exp_configs['training_params']['criterion_params'])

    def _get_batch_data(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device).float()
        y = y.to(self.device).float()
        if self.args.binary_labels:
            y = (y > self.exp_configs['eval_params']['eval_threshold']).float()
        return x, y
    
    def _one_batch_step(self, batch, batch_idx, split='train'):
        # Get batch data
        x, y = self._get_batch_data(batch, batch_idx)
        # Forward pass
        y_pred = self.model(x)  # must use forward, (e.g. predict() will not work), otherwise back propagation doesn't work.
        # Loss
        loss = self.criterion_fn(y_pred, y)

        return {'loss': loss}


#######################################################
# Evaluation function
#######################################################
    
def evaluate_piecewise(lightning_module, datamodule):
    # This loads the best model and evaluates it on the test dataloaders (one for each test piece).
    update_log_file(lightning_module.path_log, "Start evaluating on test pieces.")

    # Load the best model
    lightning_module.model.load_state_dict(torch.load(lightning_module.path_model))
    lightning_module.model.to(lightning_module.device)
    lightning_module.model.eval()

    # Get the test dataloaders
    test_dataloader_list = datamodule.get_test_dataloaders_piecewise()

    # Initialise the results
    results_df = []

    for file_idx, test_loader in enumerate(test_dataloader_list):
        fn = datamodule.splits_target['test'][file_idx]
        message = "Evaluating on test piece {}".format(fn)
        print(message)
        update_log_file(lightning_module.path_log, message)

        y_targ_all, y_pred_all = [], []

        for i, batch in enumerate(test_loader):
            print("Testing batch {}/{}".format(i+1, len(test_loader)), end='\r')

            # Get the batch data
            batch_data = lightning_module._get_batch_data(batch, i)
            x = batch_data[0]
            y = batch_data[1]

            # Predict
            y_pred = lightning_module.model.predict(x)
            
            # Update y_targ_all and y_pred_all
            y_targ = torch.squeeze(torch.squeeze(y, 2), 1).cpu().detach().numpy()
            y_pred = torch.squeeze(torch.squeeze(y_pred, 2), 1).cpu().detach().numpy()
            y_targ_all.append(y_targ)
            y_pred_all.append(y_pred)

        print()

        # Get y_targ and y_pred for the test piece
        y_targ = np.concatenate(y_targ_all, axis=0)
        y_pred = np.concatenate(y_pred_all, axis=0)
        assert y_targ.shape == y_pred.shape, "y_targ and y_pred shapes do not match. y_targ.shape: {}, y_pred.shape: {}".format(y_targ.shape, y_pred.shape)

        # Evaluate
        eval_dict = calculate_eval_measures(
            y_targ, y_pred, 
            measures=lightning_module.exp_configs['eval_params']['eval_measures'],
            threshold=lightning_module.exp_configs['eval_params']['eval_threshold'],
            save_roc_plot=False
        )
        results_df.append({'fn': fn, **eval_dict, 'n_frames': y_targ.shape[0]})

        # Update the log file
        update_log_file(lightning_module.path_log, 'File {} tested. Avarage Precision Score: {:.6f}'.format(fn, eval_dict['average_precision_score']))

    # Save the results
    results_df = pd.DataFrame(results_df)
    results_df.to_csv(lightning_module.path_result, index=False)

    # Average results
    update_log_file(lightning_module.path_log, "Average results over all test files:")
    # Average the results_df, excluding the 'fn' column
    results_df = results_df.drop('fn', axis=1)
    results_df_avg = results_df.mean(axis=0)
    # weighted average of the results_df, using n_frames as weights
    results_df_weighted_avg = results_df.iloc[:, :].multiply(results_df['n_frames'], axis=0).sum(axis=0) / results_df['n_frames'].sum()
    
    update_log_file(lightning_module.path_log, 'Average results (piece-wise average): \n{}'.format(results_df_avg))
    update_log_file(lightning_module.path_log, 'Weighted average results (frame-wise average): \n{}'.format(results_df_weighted_avg))
