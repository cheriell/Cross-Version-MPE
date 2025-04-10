import warnings
warnings.filterwarnings("ignore")

import argparse
import os, sys
sys.path.insert(0, os.path.join(sys.path[0], '../..'))
import pytorch_lightning as pl
import json
from torchinfo import summary
import torch
torch.set_float32_matmul_precision('medium')
torch.autograd.set_detect_anomaly(True)
import numpy as np
import pandas as pd
import wandb
wandb.login()

from experiments.models.aoe_cnns import aoe_cnn_model, aoe_cnn_model_no_recon
from experiments.main.utils import (
    BaseLightningModule,
    get_criterion_fn,
    path_from_config,
    update_log_file,
)
from experiments.data_loaders.aoe_dataloaders import AoEDataModule
from experiments.eval_metrics import calculate_eval_measures
from experiments.feature_preparation.utils import get_version_pairs_fn


#######################################################
# Main
#######################################################

def main(args):

    # Get the experiment configuration
    exp_configs = json.load(open(args.config_json_file, 'r'))
    path_log = path_from_config(exp_configs['path_configs']['path_log'], args)

    # Get lightning module and datamodule
    if args.task == 'domain_adaptation':
        lightning_module = aoe_cnn_lightning_module(exp_configs, args)
        datamodule = AoEDataModule(exp_configs, args)

    elif args.task == 'teacher_student':
        lightning_module = aoets_cnn_lightning_module(exp_configs, args)
        version_pairs = pd.read_csv(get_version_pairs_fn(args.workspace, exp_configs['dataset'])) if args.pick_pairs else None
        datamodule = AoEDataModule(exp_configs, args, version_pairs)

    else:
        raise ValueError("Invalid task: {}".format(args.task))

    # Train the model
    if not args.eval_only:
        update_log_file(path_log, "Start training, max_epochs: {}".format(exp_configs['training_params']['max_epochs']))

        os.makedirs(os.path.join(args.workspace, 'wandb_logs'), exist_ok=True)
        logger = pl.loggers.WandbLogger(
            project='cross-version-mpe',
            name='-'.join([exp_configs['dataset'], exp_configs['method'], args.runname]),
            save_dir=os.path.join(args.workspace, 'wandb_logs'),
        )
        trainer = pl.Trainer(
            logger=logger,
            precision=16,  # Use mixed precision
            max_epochs=exp_configs['training_params']['max_epochs'],
            devices=exp_configs['devices'],
            log_every_n_steps=50,
            reload_dataloaders_every_n_epochs=1,
        )
        trainer.fit(lightning_module, datamodule)

    # Evaluate the model
    # set the lightning model device to gpu
    lightning_module.to('cuda:{}'.format(exp_configs['devices'][0]))
    evaluate_piecewise(lightning_module, datamodule)

    wandb.finish()


#######################################################
# PyTorch Lightning Module
#######################################################

class aoe_cnn_lightning_module(BaseLightningModule):

    def __init__(self, exp_configs, args):
        """This is the PyTorch Lightning Module for the AoECNN model."""
        super().__init__(exp_configs, args)

        ae_layers = exp_configs['model_params']['ae_layers']
        self.model = aoe_cnn_model(ae_layers=ae_layers)
        # Load the pretrained model
        if args.resume_training:
            path_model = path_from_config(exp_configs['path_configs']['path_model'], args)
            if os.path.exists(path_model):
                self.model.load_state_dict(torch.load(path_model))
                update_log_file(self.path_log, "Model loaded from: {}".format(path_model))
            else:
                raise ValueError("The model file does not exist: {}".format(path_model))
        self.exp_configs = exp_configs
        self.args = args

        # Update the Log file
        update_log_file(self.path_log, "Model: {}".format(self.model.__class__.__name__))
        update_log_file(self.path_log, "\n{}".format(str(summary(self.model, input_size=(1, 6, 174, 216)))))

        # Loss functions and weights
        self.criterion_fn_mpe = get_criterion_fn(self.exp_configs['training_params']['criterion_mpe_params'])
        self.criterion_fn_recon = get_criterion_fn(self.exp_configs['training_params']['criterion_recon_params'])
        self.loss_weight_mpe_train = self.exp_configs['training_params']['loss_weights_train']['mpe']
        self.loss_weight_recon_train = self.exp_configs['training_params']['loss_weights_train']['recon']
        self.loss_weight_mpe_val = self.exp_configs['training_params']['loss_weights_val']['mpe']
        self.loss_weight_recon_val = self.exp_configs['training_params']['loss_weights_val']['recon']

    def _get_batch_data(self, batch, batch_idx):
        # Get the batch data
        x, y, y_recon, mask_mpe, mask_recon = batch
        # x
        x = x.to(self.device).float()
        # y
        y = y.to(self.device).float()
        # y_recon
        y_recon = y_recon.to(self.device).float()
        # mask_mpe
        mask_mpe = mask_mpe.to(self.device).float()
        # mask_recon
        mask_recon = mask_recon.to(self.device).float()

        return x, y, y_recon, mask_mpe, mask_recon

    def _one_batch_step(self, batch, batch_idx, split='train'):
        # Get the batch data
        x, y, y_recon, mask_mpe, mask_recon = self._get_batch_data(batch, batch_idx)
        
        # Forward pass
        y_recon_pred, y_pred = self.model(x)

        # Loss (NOTE: since we used BCEWithLogitsLoss, we cannot simply multiply the mask with the ys. sigmoid(0) = 0.5)
        loss_mpe, loss_recon = 0, 0
        count_mpe, count_recon = 0, 0
        for i in range(x.shape[0]):
            if mask_mpe[i] > 0:
                loss_mpe += self.criterion_fn_mpe(y_pred[i], y[i])
                count_mpe += 1
            if mask_recon[i] > 0:
                loss_recon += self.criterion_fn_recon(y_recon_pred[i], y_recon[i])
                count_recon += 1
        loss_mpe = loss_mpe / count_mpe
        loss_recon = loss_recon / count_recon
        if split == 'train':
            loss = self.loss_weight_mpe_train * loss_mpe + self.loss_weight_recon_train * loss_recon
        elif split == 'val':
            loss = self.loss_weight_mpe_val * loss_mpe + self.loss_weight_recon_val * loss_recon

        return {
            'loss': loss,
            'loss_mpe': loss_mpe,
            'loss_recon': loss_recon,
        }
    

#######################################################
# PyTorch Lightning Module for Teacher-Student training
#######################################################
    
class aoets_cnn_lightning_module(BaseLightningModule):

    def __init__(self, exp_configs, args):
        super().__init__(exp_configs, args)

        # ae_layers = exp_configs['model_params']['ae_layers']
        self.model = aoe_cnn_model_no_recon()
        # Load the pretrained model
        if args.resume_training:
            path_model = path_from_config(exp_configs['path_configs']['path_model'], args)
            if os.path.exists(path_model):
                self.model.load_state_dict(torch.load(path_model))
                update_log_file(self.path_log, "Model loaded from: {}".format(path_model))
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
    
def evaluate_piecewise(lightning_module : aoe_cnn_lightning_module, datamodule : AoEDataModule):
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


#######################################################
# Main (input arguments)
#######################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and evaluate a AoE-CNN model.')

    parser.add_argument('--runname', type=str, required=True, help='Name of the run')
    parser.add_argument('--config_json_file', type=str, required=True, help='Path to the config file')
    parser.add_argument('--workspace', type=str, required=True, help='Path to the workspace')

    # Input and output data paths
    parser.add_argument('--segmented_features_path_maestro', type=str, default=None)
    parser.add_argument('--segmented_features_path_schubert', type=str, default=None)
    parser.add_argument('--segmented_features_path_wagner', type=str, default=None)

    parser.add_argument('--hcqt_folder', type=str, default=None)
    parser.add_argument('--pitch_folder', type=str, default=None)
    parser.add_argument('--hcqt_folder_wagner_test', type=str, default=None)
    parser.add_argument('--pitch_folder_wagner_test', type=str, default=None)

    # Evaluation
    parser.add_argument('--eval_only', action='store_true', help='If true, only evaluate the model.')

    # Resume training
    parser.add_argument('--resume_training', action='store_true', help='If true, resume training from the last checkpoint.')


    #######################################################
    # Subparsers
    #######################################################
    subparsers = parser.add_subparsers(dest='task')

    # Domain adaptation
    parser_da = subparsers.add_parser('domain_adaptation', help='Domain adaptation')

    # Teacher student training
    parser_ts = subparsers.add_parser('teacher_student', help='Teacher student training')

    # Pretrained model
    parser_ts.add_argument('--path_pretrained_model', type=str, help='Path to the pretrained model')
    # Pseudo labels
    parser_ts.add_argument('--path_pseudo_labels', type=str, help='Path to the pseudo labels')
    parser_ts.add_argument('--binary_labels', action='store_true', help='If ture, use binarized pseudo labels (by eval_threshold) for training and validation. Else, use the probability values as target labels.')
    # Cross-version training
    parser_ts.add_argument('--path_valid_frames', type=str, help='Path to the valid frames')
    parser_ts.add_argument('--pick_pairs', action='store_true', help='If true, pick from version pairs for training and validation. This only applies to when there are different pseudo labels obtained from different version pairs of the same piece.')

    args = parser.parse_args()

    main(args)