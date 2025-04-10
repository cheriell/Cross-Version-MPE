import warnings
warnings.filterwarnings("ignore")

import argparse
import os, sys
sys.path.insert(0, os.path.join(sys.path[0], '../..'))
import json
import torch
torch.backends.cudnn.benchmark = True  # Improves the performance of the model if input size is constant
from torchinfo import summary
import numpy as np
import pandas as pd

from experiments.models.basic_cnns import deep_cnn_segm_sigmoid
from experiments.main.utils import Experiment, path_from_config
from experiments.data_loaders.mpe_dataloaders import get_mpe_dataloader
from experiments.eval_metrics import calculate_eval_measures
from experiments.feature_preparation.utils import get_version_pairs_fn

def main(args):

    #######################################################
    # Experiment setup
    #######################################################

    # Get experiment configurations
    exp_configs = json.load(open(args.config_json_file, 'r'))
    min_pitch = exp_configs['model_params']['min_pitch']
    n_bins_out = exp_configs['model_params']['n_bins_out']
    
    # Start the experiment
    exp = Experiment(exp_configs, args)

    # Get the model
    model = deep_cnn_segm_sigmoid(
        n_chan_input=exp_configs['model_params']['n_chan_input'],
        n_chan_layers=exp_configs['model_params']['n_chan_layers'],
        n_prefilt_layers=exp_configs['model_params']['n_prefilt_layers'],
        residual=exp_configs['model_params']['residual'],
        n_bins_in=exp_configs['model_params']['n_bins_in'],
        n_bins_out=exp_configs['model_params']['n_bins_out'],
        a_lrelu=exp_configs['model_params']['a_lrelu'],
        p_dropout=exp_configs['model_params']['p_dropout'],
    )

    exp.log('Model: {}'.format(model.__class__.__name__))
    exp.log('\n{}'.format(str(summary(model, input_size=(1, 6, 174, 216)))))

    # Get the training, validation, and testing data loaders
    splits = json.load(open('dataset_splits/{}_split.json'.format(exp_configs['dataset']), 'r'))
    
    #######################################################
    # Model training
    #######################################################
    if not args.eval_only:
        # Get the data loaders
        path_x = path_from_config(exp_configs['path_configs']['path_x'], args)
        path_y = path_from_config(exp_configs['path_configs']['path_y'], args)

        train_loader = get_mpe_dataloader(
            fns=splits['train'], 
            path_x=path_x, 
            path_y=path_y, 
            dataset_params=exp_configs['train_dataset_params'], 
            dataloader_params=exp_configs['train_loader_params'], 
            path_valid_frames=path_from_config(exp_configs['path_configs']['path_valid_frames'], args), 
            eval=False,
            pick_pairs=args.pick_pairs,
            version_pairs=pd.read_csv(get_version_pairs_fn(args.workspace, exp_configs['dataset'])) if args.pick_pairs else None,
        )
        val_loader = get_mpe_dataloader(
            fns=splits['val'], 
            path_x=path_x, 
            path_y=path_y, 
            dataset_params=exp_configs['val_dataset_params'], 
            dataloader_params=exp_configs['val_loader_params'], 
            path_valid_frames=path_from_config(exp_configs['path_configs']['path_valid_frames'], args),
            eval=True,
            pick_pairs=args.pick_pairs,
            version_pairs=pd.read_csv(get_version_pairs_fn(args.workspace, exp_configs['dataset'])) if args.pick_pairs else None,
        )

        exp.log('Train loader length: {}'.format(len(train_loader)))
        exp.log('Val loader length: {}'.format(len(val_loader)))

        # Prepare training (criterion, optimizer, scheduler, early stopping)
        exp.prepare_training(
            model=model,
            criterion_params=exp_configs['training_params']['criterion_params'],
            optimizer_params=exp_configs['training_params']['optimizer_params'],
            scheduler_params=exp_configs['training_params']['scheduler_params'],
            early_stopping_params=exp_configs['training_params']['early_stopping_params']
        )

        exp.log('Start training, max_epochs = {}'.format(exp_configs['training_params']['max_epochs']))

        for epoch in range(exp_configs['training_params']['max_epochs']):

            # Train
            accum_loss, n_batches = 0, 0
            for i, (x, y) in enumerate(train_loader):
                print('Epoch: {}, Training Batch: {}/{}'.format(epoch, i, len(train_loader)), end='\r')
                # Get the data
                x = x.to(exp.device).float()
                if y.shape[3] == 128:
                    y = y.to(exp.device).float()[:,:,:,min_pitch:min_pitch+n_bins_out]
                elif y.shape[3] == n_bins_out:
                    y = y.to(exp.device).float()
                else:
                    raise ValueError('y shape not recognized. y shape: {}'.format(y.shape))
                if args.binary_labels:
                    y = (y > exp_configs['eval_params']['eval_threshold']).float()
                # forward pass
                y_pred = model(x)
                loss = exp.criterion_fn(y_pred, y)
                # backward pass
                exp.optimizer.zero_grad()
                loss.backward()
                exp.optimizer.step()
                # accumulate loss
                accum_loss += loss.item()
                n_batches += 1
            print()
            train_loss = accum_loss / n_batches

            # Validate
            accum_loss_val, n_batches_val = 0, 0
            for i, (x, y) in enumerate(val_loader):
                print('Epoch: {}, Validation Batch: {}/{}'.format(epoch, i, len(val_loader)), end='\r')
                # Get the data
                x = x.to(exp.device).float()
                if y.shape[3] == 128:
                    y = y.to(exp.device).float()[:,:,:,min_pitch:min_pitch+n_bins_out]
                elif y.shape[3] == n_bins_out:
                    y = y.to(exp.device).float()
                else:
                    raise ValueError('y shape not recognized. y shape: {}'.format(y.shape))
                if args.binary_labels:
                    y = (y > exp_configs['eval_params']['eval_threshold']).float()
                # forward pass
                y_pred = model(x)
                loss = exp.criterion_fn(y_pred, y)
                # accumulate loss
                accum_loss_val += loss.item()
                n_batches_val += 1
            print()
            val_loss = accum_loss_val / n_batches_val

            # Epoch end
            exp.epoch_end(train_loss, val_loss)

    #######################################################
    # Model evaluation
    #######################################################
    exp.log('Start testing...')

    # Load the best model
    model.load_state_dict(torch.load(exp.path_model))
    model.to(exp.device)
    model.eval()

    # Get the test data loaders
    path_x_test = path_from_config(exp_configs['path_configs']['path_x_test'], args)
    path_y_test = path_from_config(exp_configs['path_configs']['path_y_test'], args)

    test_loaders = [get_mpe_dataloader(
        fns=[fn], 
        path_x=path_x_test, 
        path_y=path_y_test, 
        dataset_params=exp_configs['test_dataset_params'], 
        dataloader_params=exp_configs['test_loader_params'],
        path_valid_frames=None,
        eval=True,
    ) for fn in splits['test']]  # list of test loaders for each test file

    exp.log('Test loaders length in total: {}'.format(np.sum([len(l) for l in test_loaders])))

    results_df = []

    # Iterate over the test files
    for file_idx, test_loader in enumerate(test_loaders):
        print('Testing file: {}/{}'.format(file_idx, len(test_loaders)))

        y_targ_all, y_pred_all = [], []

        for i, (x, y) in enumerate(test_loader):
            print('Testing Batch: {}/{}'.format(i, len(test_loader)), end='\r')
            # Get the data
            x = x.to(exp.device).float()
            if y.shape[3] == 128:
                y = y.to(exp.device).float()[:,:,:,min_pitch:min_pitch+n_bins_out]
            elif y.shape[3] == n_bins_out:
                y = y.to(exp.device).float()
            else:
                raise ValueError('y shape not recognized. y shape: {}'.format(y.shape))
            if args.binary_labels:
                y = (y > exp_configs['eval_params']['eval_threshold']).float()
            # predict
            y_pred = model(x)
            y_targ = torch.squeeze(torch.squeeze(y, 2), 1).cpu().detach().numpy()
            y_pred = torch.squeeze(torch.squeeze(y_pred, 2), 1).cpu().detach().numpy()
            y_targ_all.append(y_targ)
            y_pred_all.append(y_pred)
        print()

        y_targ = np.concatenate(y_targ_all, axis=0)
        y_pred = np.concatenate(y_pred_all, axis=0)

        assert y_targ.shape == y_pred.shape, "y_targ and y_pred shapes do not match. y_targ shape: {}, y_pred shape: {}".format(y_targ.shape, y_pred.shape)

        # Evaluate
        eval_dict = calculate_eval_measures(
            y_targ, y_pred, 
            measures=exp_configs['eval_params']['eval_measures'],
            threshold=exp_configs['eval_params']['eval_threshold'],
            save_roc_plot=False
        )
        results_df.append({'fn': splits['test'][file_idx], **eval_dict, 'n_frames': y_targ.shape[0]})

        exp.log('File {} tested. Avarage Precision Score: {:.6f}'.format(splits['test'][file_idx], eval_dict['average_precision_score']))

    # Save and log the results
    results_df = pd.DataFrame(results_df)
    results_df.to_csv(exp.path_result, index=False)

    exp.log('Average results over all test files...')

    # Average the results_df, excluding the 'fn' column
    results_df = results_df.drop('fn', axis=1)
    results_df_avg = results_df.mean(axis=0)
    # weighted average of the results_df, using n_frames as weights
    results_df_weighted_avg = results_df.iloc[:, :].multiply(results_df['n_frames'], axis=0).sum(axis=0) / results_df['n_frames'].sum()

    exp.log('Average results (piece-wise average): \n{}'.format(results_df_avg))
    exp.log('Weighted average results (frame-wise average): \n{}'.format(results_df_weighted_avg))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and evaluate a Res-CNN model.')

    parser.add_argument('--runname', type=str, required=True, help="Name of the run.")
    parser.add_argument('--config_json_file', type=str, required=True, help="Path to the config json file.")
    parser.add_argument('--workspace', type=str, required=True)
    
    # Input and output paths
    parser.add_argument('--segmented_features_path_maestro', type=str, default=None)
    parser.add_argument('--segmented_features_path_schubert', type=str, default=None)
    parser.add_argument('--segmented_features_path_wagner', type=str, default=None)
    parser.add_argument('--hcqt_folder', type=str, default=None)
    parser.add_argument('--pitch_folder', type=str, default=None)
    parser.add_argument('--hcqt_folder_wagner_test', type=str, default=None)
    parser.add_argument('--pitch_folder_wagner_test', type=str, default=None)

    # Pseudo labels
    parser.add_argument('--path_pseudo_labels', type=str, default=None)
    parser.add_argument('--binary_labels', action='store_true', help='If true, use binarized target labels (by eval_threshold) for training and validaqtion. Else, use the probability values as target labels.')

    # Cross-version training
    parser.add_argument('--pick_pairs', action='store_true', help='If true, pick version pairs for cross-version training (load annotations from <fn>_by_alignment_<fn_aligned>), when there are different pseudo labels generated from each version pair.')
    
    # Evaluation
    parser.add_argument('--eval_only', action='store_true', help="If true, only evaluate the model without training. Need to specify the model path in config file.")

    args = parser.parse_args()

    main(args)

