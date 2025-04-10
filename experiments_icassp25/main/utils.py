import os
import json
from datetime import datetime
import torch
import numpy as np
import pytorch_lightning as pl
from torchinfo import summary


#######################################################
# Experiment class (used to manage the training of the ResCNN model)
#######################################################

class Experiment(object):
    """Experiment class. This is used to manage the training of the ResCNN model.
    """

    def __init__(self, exp_configs, args):
        super(Experiment, self).__init__()
        print('Initializing experiment...')

        # Paths to the experimental outputs
        self.path_model = path_from_config(exp_configs['path_configs']['path_model'], args)
        self.path_result = path_from_config(exp_configs['path_configs']['path_result'], args)
        self.path_log = path_from_config(exp_configs['path_configs']['path_log'], args)
        self.path_config = path_from_config(exp_configs['path_configs']['path_config'], args)

        # Create directories
        os.makedirs(os.path.dirname(self.path_model), exist_ok=True)
        os.makedirs(os.path.dirname(self.path_result), exist_ok=True)
        os.makedirs(os.path.dirname(self.path_log), exist_ok=True)
        os.makedirs(os.path.dirname(self.path_config), exist_ok=True)
        
        # Empty the log file if it's not eval_only
        if not args.eval_only:
            open(self.path_log, 'w').close()
        # Save the config file
        json.dump(exp_configs, open(self.path_config, 'w'), indent=4)

        # Set the device
        self.device = exp_configs['device']

    def log(self, message):
        update_log_file(self.path_log, message)

    def prepare_training(self, model, criterion_params, optimizer_params, scheduler_params, early_stopping_params):
        self.model = model
        self.model.to(self.device)

        self.criterion_fn = get_criterion_fn(criterion_params)
        self.optimizer = get_optimizer(model, optimizer_params)
        self.scheduler = get_scheduler(self.optimizer, scheduler_params)
        self.early_stopping_callback = get_early_stopping_callback(early_stopping_params)

    def epoch_end(self, train_loss, val_loss):
        # Get current learning rate
        if self.scheduler.__class__.__name__ == 'LambdaLR':
            lr_cur = self.scheduler.get_last_lr()[0]
        elif self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            lr_cur = self.optimizer.param_groups[0]['lr']
        else:
            raise ValueError('Scheduler not recognized.')
        # Get current epoch from scheduler
        self.epoch = self.scheduler.last_epoch
        
        # log the results
        self.log('Epoch: {} | Train loss: {:.6f} | Val loss: {:.6f} | Learning rate: {:.6f}'.format(self.epoch, train_loss, val_loss, lr_cur))

        # Scheduler step
        if self.scheduler.__class__.__name__ == 'LambdaLR':
            self.scheduler.step()
        elif self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            self.scheduler.step(val_loss)
        else:
            raise ValueError('Scheduler not recognized.')
        
        # Early stopping
        if self.epoch == 0:
            torch.save(self.model.state_dict(), self.path_model)
            self.log('Model saved at epoch {}'.format(self.epoch))
        else:
            if self.early_stopping_callback.curr_is_better(val_loss):
                torch.save(self.model.state_dict(), self.path_model)
                self.log('Model saved at epoch {}'.format(self.epoch))
        self.early_stopping_callback.step(val_loss)


#######################################################
# Base lightning module
#######################################################

class BaseLightningModule(pl.LightningModule):

    def __init__(self, exp_configs, args):
        super().__init__()

        # Paths to the experimental outputs
        self.path_model = path_from_config(exp_configs['path_configs']['path_model'], args)
        self.path_result = path_from_config(exp_configs['path_configs']['path_result'], args)
        self.path_log = path_from_config(exp_configs['path_configs']['path_log'], args)
        self.path_config = path_from_config(exp_configs['path_configs']['path_config'], args)

        # Create directories
        os.makedirs(os.path.dirname(self.path_model), exist_ok=True)
        os.makedirs(os.path.dirname(self.path_result), exist_ok=True)
        os.makedirs(os.path.dirname(self.path_log), exist_ok=True)
        os.makedirs(os.path.dirname(self.path_config), exist_ok=True)

        # Empty the log file if it's not eval_only
        if not args.eval_only:
            open(self.path_log, 'w').close()
        # Save the config file
        json.dump(exp_configs, open(self.path_config, 'w'), indent=4)

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = get_optimizer(self.model, self.exp_configs['training_params']['optimizer_params'])
        scheduler = get_scheduler(optimizer, self.exp_configs['training_params']['scheduler_params'])
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss',
        }
    
    def configure_callbacks(self):
        return get_callbacks_list(self.path_model, self.path_log, self.exp_configs['training_params']['early_stopping_params'])
    
    def _get_batch_data(self, batch, batch_idx):
        raise NotImplementedError
    
    def _one_batch_step(self, batch, batch_idx, split='train'):
        raise NotImplementedError
    
    def training_step(self, batch, batch_idx):
        metrics = self._one_batch_step(batch, batch_idx, split='train')
        for key, value in metrics.items():
            self.log('train_{}'.format(key), value, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': metrics['loss'], 'logs': metrics}

    def validation_step(self, batch, batch_idx):
        metrics = self._one_batch_step(batch, batch_idx, split='val')
        for key, value in metrics.items():
            self.log('val_{}'.format(key), value, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': metrics['loss'], 'logs': metrics}
    

#######################################################
# Helper functions
#######################################################

def path_from_config(path_config, args):
    path =  path_config.format(
        workspace=args.workspace,
        runname=args.runname,
        segmented_features_path_maestro=args.segmented_features_path_maestro if hasattr(args, 'segmented_features_path_maestro') else None,
        segmented_features_path_schubert=args.segmented_features_path_schubert if hasattr(args, 'segmented_features_path_schubert') else None,
        segmented_features_path_wagner=args.segmented_features_path_wagner if hasattr(args, 'segmented_features_path_wagner') else None,
        hcqt_folder=args.hcqt_folder if hasattr(args, 'hcqt_folder') else None,
        pitch_folder=args.pitch_folder if hasattr(args, 'pitch_folder') else None,
        hcqt_folder_wagner_test=args.hcqt_folder_wagner_test if hasattr(args, 'hcqt_folder_wagner_test') else None,
        pitch_folder_wagner_test=args.pitch_folder_wagner_test if hasattr(args, 'pitch_folder_wagner_test') else None,
        path_pseudo_labels=args.path_pseudo_labels if hasattr(args, 'path_pseudo_labels') else None,
        path_valid_frames=args.path_valid_frames if hasattr(args, 'path_valid_frames') else None,
    )
    path = path.replace('//', '/')
    return path

def update_log_file(path_log, message):
    with open(path_log, 'a') as f:
        f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' | INFO:\n')
        f.write(message + '\n\n')
    print("LOG:", message)

def get_criterion_fn(criterion_params):
    if criterion_params['name'] == 'binary_cross_entropy':
        criterion_fn = torch.nn.BCELoss(reduction=criterion_params['reduction'])
    elif criterion_params['name'] == 'binary_cross_entropy_with_logits':
        criterion_fn = torch.nn.BCEWithLogitsLoss(reduction=criterion_params['reduction'])
    elif criterion_params['name'] == 'l1_loss':
        criterion_fn = torch.nn.L1Loss(reduction=criterion_params['reduction'])
    elif criterion_params['name'] == 'mse_loss':
        criterion_fn = torch.nn.MSELoss(reduction=criterion_params['reduction'])
    else:
        raise ValueError('Loss function not recognized.')

    return criterion_fn

def get_optimizer(model, optimizer_params):
    if optimizer_params['name']=='SGD':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=optimizer_params['initial_lr'], 
            momentum=optimizer_params['momentum']
        )
    elif optimizer_params['name']=='Adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=optimizer_params['initial_lr'], 
            betas=optimizer_params['betas']
        )
    elif optimizer_params['name']=='AdamW':
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=optimizer_params['initial_lr'], 
            betas=optimizer_params['betas'], 
            eps=optimizer_params['eps'], 
            weight_decay=optimizer_params['weight_decay'], 
            amsgrad=optimizer_params['amsgrad']
        )
    else:
        raise ValueError('Optimizer not recognized.')
    return optimizer

def get_scheduler(optimizer, scheduler_params):
    if not scheduler_params['use_scheduler']:
        raise ValueError('Scheduler not used.')

    if scheduler_params['name']=='LambdaLR':
        
        start_lr = scheduler_params['start_lr']
        end_lr = scheduler_params['end_lr']
        n_decay = scheduler_params['n_decay']
        exp_decay = scheduler_params['exp_decay']

        polynomial_decay = lambda epoch: ((start_lr - end_lr) * (1 - min(epoch, n_decay)/n_decay) ** exp_decay ) + end_lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=polynomial_decay)

    elif scheduler_params['name']=='ReduceLROnPlateau':

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode=scheduler_params['mode'], 
            factor=scheduler_params['factor'], 
            patience=scheduler_params['patience'], 
            threshold=scheduler_params['threshold'], 
            threshold_mode=scheduler_params['threshold_mode'], 
            cooldown=scheduler_params['cooldown'], 
            eps=scheduler_params['eps'], 
            min_lr=scheduler_params['min_lr'], 
            verbose=scheduler_params['verbose']
        )

    else:
        raise ValueError('Scheduler not recognized.')
    
    return scheduler

def get_early_stopping_callback(early_stopping_params):
    # This will return the early stopping callback implemented by Christof.
    if not early_stopping_params['use_early_stopping']:
        raise ValueError('Early stopping not used.')
    
    early_stopping_callback = EarlyStopping(
        mode=early_stopping_params['mode'], 
        min_delta=early_stopping_params['min_delta'], 
        patience=early_stopping_params['patience'], 
        percentage=early_stopping_params['percentage']
    )
    return early_stopping_callback

def get_callbacks_list(path_model, path_log, early_stopping_params):
        epoch_end_callback = EpochEndCallback(path_log, path_model)
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor='val_loss_epoch',
            patience=early_stopping_params['patience'],
            mode=early_stopping_params['mode'],
            min_delta=early_stopping_params['min_delta'],
        )
        return [epoch_end_callback, early_stopping_callback]


#######################################################
# Early stopping callback (copied from Christof's public code, used in the ResCNN model)
#######################################################

class EarlyStopping(object):
    """ Early stopping class.

        Default parameters:
            mode='min'          lower value of metric is better (e.g. for loss)
            min_delta=0         margin by which metric has to improve
            patience=10         number of epochs to wait for improvement
            percentage=False    inidcates if min_delta is absolute or relative

    """
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)

    def curr_is_better(self, metrics):
        return self.is_better(metrics, self.best)


#######################################################
# Epoch end callback
#######################################################
    
class EpochEndCallback(pl.Callback):
    """Callback to be used at the end of each epoch.
    This will update the log file with the training loss, validation loss, and the current learning rate.
    """

    def __init__(self, path_log, path_model):
        self.path_log = path_log
        self.path_model = path_model

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get the current epoch
        epoch = trainer.current_epoch

        # Get the training and validation losses
        logged_metrics = trainer.callback_metrics

        # Get current learning rate
        lr_cur = trainer.optimizers[0].param_groups[0]['lr']

        # Log the training and validation losses
        update_log_file(self.path_log, 'Epoch: {} | Learning rate: {:.6f}'.format(epoch, lr_cur))
        update_log_file(self.path_log, 'Logged metrics: {}'.format(logged_metrics))

        # Save the best model
        if epoch == 0:
            torch.save(pl_module.model.state_dict(), self.path_model)
            update_log_file(self.path_log, 'Model saved at epoch {}'.format(epoch))
            self.val_loss_epoch = logged_metrics['val_loss_epoch']
        elif logged_metrics['val_loss_epoch'] < self.val_loss_epoch:
                torch.save(pl_module.model.state_dict(), self.path_model)
                update_log_file(self.path_log, 'Model saved at epoch {}'.format(epoch))
                self.val_loss_epoch = logged_metrics['val_loss_epoch']
        else:
            pass

