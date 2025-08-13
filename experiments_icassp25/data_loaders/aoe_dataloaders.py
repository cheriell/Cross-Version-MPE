import pytorch_lightning as pl
import json
import os
import torch
import numpy as np

from experiments_icassp25.feature_preparation.utils import (
    get_feature_segment_metainfo_fn,
    extend_fn,
)
from experiments_icassp25.data_loaders.utils import (
    load_sample_from_segmented_feature,
    data_augmentation,
    HalfHalfSampler,
)
from experiments_icassp25.main.utils import path_from_config, update_log_file
from experiments_icassp25.models.aoe_cnns import min_pitch, n_bins_out


##################################################
# AoEDataModule (for the AoE model)
##################################################

class AoEDataModule(pl.LightningDataModule):

    def __init__(self, exp_configs, args, version_pairs=None):
        """
        Parameters
        ----------
        exp_configs : dict
            The experiment configurations from the config file.
        args : argparse.Namespace
            The input arguments from the main script.
        version_pairs : pd.DataFrame
            The version pairs for cross-version training.
        """
        super().__init__()

        self.dataset_str = exp_configs['dataset']
        self.data_params = exp_configs['data_params']
        self.path_configs = exp_configs['path_configs']
        self.pick_pairs = args.pick_pairs if hasattr(args, 'pick_pairs') else False
        self.version_pairs = version_pairs
        self.args = args
        self.path_log = path_from_config(self.path_configs['path_log'], self.args)

        # Get the dataset splits
        self.splits_source = json.load(open('dataset_splits/maestro_split.json', 'r'))
        self.splits_target = json.load(open('dataset_splits/{}_split.json'.format(self.dataset_str), 'r'))
        
        update_log_file(self.path_log, "Datamodule setup complete.")

    def train_dataloader(self):

        if self.args.task == 'domain_adaptation':
            dataset_source_list, dataset_target_list = [], []
            for fn in self.splits_source['train']:
                dataset_source_list.append(aoe_dataset_source(
                    fn=fn,
                    path_x=path_from_config(self.path_configs['path_x_source'], self.args),
                    path_y=path_from_config(self.path_configs['path_y_source'], self.args),
                    dataset_params=self.data_params['train_dataset_params'],
                    eval=False,
                ))
            for fn in self.splits_target['train']:
                dataset_target_list.append(aoe_dataset_target(
                    fn=fn,
                    path_x=path_from_config(self.path_configs['path_x_target'], self.args),
                    path_y=None,
                    dataset_params=self.data_params['train_dataset_params'],
                    task='domain_adaptation',
                    eval=False,
                    predict=False,
                ))
            dataset_source = torch.utils.data.ConcatDataset(dataset_source_list)
            dataset_target = torch.utils.data.ConcatDataset(dataset_target_list)
            dataset = torch.utils.data.ConcatDataset([dataset_source, dataset_target])

            # Use costom random sampler (half source, half target)
            sampler = HalfHalfSampler(len(dataset_source), len(dataset_target), shuffle=True)

            # Get DataLoader
            dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, **self.data_params['train_loader_params'])

            # Update the log file
            update_log_file(self.path_log, "Train dataloader setup complete.\n \
                len(dataset_source): {}, len(dataset_target): {} \n \
                Train dataloader length: {}".format(len(dataset_source), len(dataset_target), len(dataloader)))

        elif self.args.task == 'teacher_student':
            # Get dataset
            dataset_list = []
            for fn in self.splits_target['train']:
                dataset_list.append(aoe_dataset_target(
                    fn=fn,
                    path_x=path_from_config(self.path_configs['path_x_train'], self.args),
                    path_y=path_from_config(self.path_configs['path_y_train'], self.args),
                    dataset_params=self.data_params['train_dataset_params'],
                    task='teacher_student',
                    path_valid_frames=path_from_config(self.path_configs['path_valid_frames'], self.args) if self.path_configs['path_valid_frames'] is not None else None,
                    pick_pairs=self.pick_pairs,
                    version_pairs=self.version_pairs,
                    eval=False,
                    predict=False,
                ))
            dataset = torch.utils.data.ConcatDataset(dataset_list)

            # Get random sampler and dataloader
            sampler = torch.utils.data.RandomSampler(dataset)
            dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, **self.data_params['train_loader_params'])

            # Update the log file
            update_log_file(self.path_log, "train dataloader setup complete.\n \
                Train dataloader length: {}".format(len(dataloader)))

        else:
            raise ValueError("Task not supported.")
        
        return dataloader

    def val_dataloader(self):
        if self.args.task == 'domain_adaptation':
            dataset_source_list, dataset_target_list = [], []
            for fn in self.splits_source['val']:
                dataset_source_list.append(aoe_dataset_source(
                    fn=fn,
                    path_x=path_from_config(self.path_configs['path_x_source'], self.args),
                    path_y=path_from_config(self.path_configs['path_y_source'], self.args),
                    dataset_params=self.data_params['val_dataset_params'],
                    eval=True,
                ))
            for fn in self.splits_target['val']:
                dataset_target_list.append(aoe_dataset_target(
                    fn=fn,
                    path_x=path_from_config(self.path_configs['path_x_target'], self.args),
                    path_y=None,
                    dataset_params=self.data_params['val_dataset_params'],
                    task='domain_adaptation',
                    eval=True,
                    predict=False,
                ))
            dataset_source = torch.utils.data.ConcatDataset(dataset_source_list)
            dataset_target = torch.utils.data.ConcatDataset(dataset_target_list)
            dataset = torch.utils.data.ConcatDataset([dataset_source, dataset_target])

            # Use half-half sampler
            sampler = HalfHalfSampler(len(dataset_source), len(dataset_target), shuffle=False)

            # Get DataLoader
            dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, **self.data_params['val_loader_params'])

            # Update the log file
            update_log_file(self.path_log, "Validation dataloader setup complete.\n \
                len(dataset_source): {}, len(dataset_target): {} \n \
                Validation dataloader length: {}".format(len(dataset_source), len(dataset_target), len(dataloader)))

        elif self.args.task == 'teacher_student':
            # Get dataset
            dataset_list = []
            for fn in self.splits_target['val']:
                dataset_list.append(aoe_dataset_target(
                    fn=fn,
                    path_x=path_from_config(self.path_configs['path_x_train'], self.args),
                    path_y=path_from_config(self.path_configs['path_y_train'], self.args),
                    dataset_params=self.data_params['val_dataset_params'],
                    task='teacher_student',
                    path_valid_frames=path_from_config(self.path_configs['path_valid_frames'], self.args) if self.path_configs['path_valid_frames'] is not None else None,
                    pick_pairs=self.pick_pairs,
                    version_pairs=self.version_pairs,
                    eval=True,
                    predict=False,
                ))
            dataset = torch.utils.data.ConcatDataset(dataset_list)
            # Use sequential sampler
            sampler = torch.utils.data.SequentialSampler(dataset)
            # Get DataLoader
            dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, **self.data_params['val_loader_params'])
            # Update the log file
            update_log_file(self.path_log, "Validation dataloader setup complete.\n \
                Validation dataloader length: {}".format(len(dataloader)))

        else:
            raise ValueError("Task not supported.")

        return dataloader
    
    def get_test_dataloaders_piecewise(self):
        # We return a list of dataloaders, one for each test piece.
        # This is used for the piecewise evaluation.
        dataloader_list = []

        for fn in self.splits_target['test']:
            # Get dataset
            dataset_target = aoe_dataset_target(
                fn=fn,
                path_x=path_from_config(self.path_configs['path_x_test'], self.args),
                path_y=path_from_config(self.path_configs['path_y_test'], self.args),
                dataset_params=self.data_params['test_dataset_params'],
                task=self.args.task,
                path_valid_frames=None,
                pick_pairs=False,
                version_pairs=None,
                eval=True,
                predict=True,
            )
            # Use sequential sampler
            sampler = torch.utils.data.SequentialSampler(dataset_target)
            # Get DataLoader
            dataloader = torch.utils.data.DataLoader(dataset_target, sampler=sampler, **self.data_params['test_loader_params'])

            dataloader_list.append(dataloader)

        # Update the log file
        update_log_file(self.path_log, "Test dataloader setup complete, total number of pieces: {}".format(len(dataloader_list)))

        return dataloader_list


##################################################
# Source dataset for the AoE model
##################################################

class aoe_dataset_source(torch.utils.data.Dataset):

    def __init__(self, fn, path_x, path_y, dataset_params, eval=False):
        """
        Parameters
        ----------
        fn : str
            The filename of the feature before segmentation.
        path_x : str
            Path to the segmented input features (hcqt).
        path_y : str
            Path to the segmented output features (pitch).
        dataset_params : dict
            The parameters for the dataset from the config file.
        eval : bool
            If true, the dataset is for evaluation (validation / testing).
        """
        self.fn = fn
        self.path_x = path_x
        self.path_y = path_y
        self.eval = eval

        self.context = dataset_params['context']
        self.stride = dataset_params['stride']
        self.compression = dataset_params['compression']

        # Data augmentation
        if not eval:
            self.augmentation = dataset_params['augmentation']

        # Get dataset length, example metainfo: {"n_segments": "125", "segment_length": "30", "length": "373166"}
        self.fn_seg_metainfo = json.load(open(os.path.join(path_x, get_feature_segment_metainfo_fn(fn)), 'r'))
        self.segment_length = int(self.fn_seg_metainfo['segment_length'])
        self.full_length = int(self.fn_seg_metainfo['length'])

        self.len = (self.full_length - self.context) // self.stride

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        # Get annotation fn
        fn_y = self.fn

        # Change dataset index to the corresponding idx in the input and output features.
        if self.eval:
            idx = index * self.stride + self.context // 2
        else:
            idx = np.random.randint(self.context // 2, self.full_length - self.context // 2)

        # Load input and output features
        x, idx_in_segment = load_sample_from_segmented_feature(idx, self.segment_length, self.context, self.path_x, self.fn, axis=1)
        y, _ = load_sample_from_segmented_feature(idx, self.segment_length, self.context, self.path_y, fn_y, axis=1)
        if y.shape[0] == 128:
            y = y[min_pitch:min_pitch + n_bins_out, :]

        # Get the context by the idx
        x_context = torch.from_numpy(x[:, idx_in_segment - self.context // 2 : idx_in_segment + self.context // 2 + 1, :])
        y_context = torch.from_numpy(y[:, idx_in_segment])
        # transpose
        x_context = x_context.permute(2, 1, 0)   # (n_chan, n_frames, n_bins)
        y_context = y_context.unsqueeze(0).unsqueeze(1)   # (1, 1, n_bins)

        # Data augmentation
        if not self.eval:
            x_context, y_context = data_augmentation(x_context, y_context, self.augmentation)

        # Reconstruction
        x_context_recon = x_context.clone()

        # Masks for the MPE backend and the reconstruction
        mask_mpe = 1  # Train with MPE
        mask_recon = 0  # Ignore the reconstruction loss

        return x_context, y_context, x_context_recon, mask_mpe, mask_recon
                

##################################################
# Target dataset for the AoE model
##################################################

class aoe_dataset_target(torch.utils.data.Dataset):

    def __init__(self, fn, path_x, path_y, dataset_params, task='domain_adaptation', path_valid_frames=None, pick_pairs=False, version_pairs=None, eval=False, predict=False):
        """
        Parameters
        ----------
        fn : str
            The filename of the feature before segmentation.
        path_x : str
            Path to the segmented input features (hcqt).
        path_y : str
            Path to the segmented output features (pitch).
        dataset_params : dict
            The parameters for the dataset from the config file.
        task : str
            The task of the model. domain_adaptation or teacher_student.
        path_valid_frames : str
            Path to the valid frames for cross-version training.
        pick_pairs : bool
            If true, pick from version pairs for training and validation. This only applies to when there are different pseudo labels obtained from different version pairs of the same piece.
        version_pairs : pd.DataFrame
            The version pairs for cross-version training.
        eval : bool
            If true, the dataset is for evaluation (validation / testing).
        predict : bool
            If true, the dataset is for prediction (testing).
        """
        self.fn = fn
        self.path_x = path_x
        self.path_y = path_y
        self.task = task
        self.path_valid_frames = path_valid_frames
        self.pick_pairs = pick_pairs
        self.eval = eval
        self.predict = predict

        self.context = dataset_params['context']
        self.stride = dataset_params['stride']
        self.compression = dataset_params['compression']

        # Data augmentation
        if not eval:
            self.augmentation = dataset_params['augmentation']

        # Prepare version pairs for cross-version training
        if pick_pairs:
            self.fn_aligned_all = set([pair['fn_aligned'] for _, pair in version_pairs.iterrows() if pair['fn1'] == fn or pair['fn2'] == fn])
            self.fn_aligned_all = list(self.fn_aligned_all)

        # Get dataset length, example metainfo: {"n_segments": "125", "segment_length": "30", "length": "373166"}
        self.fn_seg_metainfo = json.load(open(os.path.join(path_x, get_feature_segment_metainfo_fn(fn)), 'r'))
        self.segment_length = int(self.fn_seg_metainfo['segment_length'])
        self.full_length = int(self.fn_seg_metainfo['length'])

        if path_valid_frames is None:
            self.len = (self.full_length - self.context) // self.stride
        else:
            if pick_pairs:
                if len(self.fn_aligned_all) == 0:
                    print('INFO: There is no version pair for the piece: {}'.format(fn))
                    self.len = 0
                    return
                else:
                    valid_frames = np.load(os.path.join(path_valid_frames, extend_fn(self.fn, fn_aligned=self.fn_aligned_all[0])))  # use the first fn_aligned to calculate the dataset length
            else:
                valid_frames = np.load(os.path.join(path_valid_frames, fn))
            valid_frames = valid_frames[valid_frames >= self.context // 2]
            valid_frames = valid_frames[valid_frames < self.full_length - self.context // 2]
            valid_frames_count = valid_frames.shape[0]

            self.len = valid_frames_count // self.stride

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        # Get annotation fn
        if self.pick_pairs:
            if self.eval:
                fn_y = extend_fn(self.fn, fn_aligned=self.fn_aligned_all[index % len(self.fn_aligned_all)])  # avoid randomness in model validation
            else:
                fn_y = extend_fn(self.fn, fn_aligned=np.random.choice(self.fn_aligned_all))  # randomly pick a version pair for training
        else:
            fn_y = self.fn

        # Change dataset index to the corresponding idx in the input and output features.
        if self.path_valid_frames is not None:
            valid_frames = np.load(os.path.join(self.path_valid_frames, fn_y))
            valid_frames = valid_frames[valid_frames >= self.context // 2]
            valid_frames = valid_frames[valid_frames < self.full_length - self.context // 2]
            if self.eval:
                idx = valid_frames[(index * self.stride) % valid_frames.shape[0]]  # Since we used the first fn_aligned to calculate the dataset length, we need to use the modulo here.
            else:
                idx = np.random.choice(valid_frames)
        else:
            if self.eval:
                idx = index * self.stride + self.context // 2
            else:
                idx = np.random.randint(self.context // 2, self.full_length - self.context // 2)

        # Load input and output features
        x, idx_in_segment = load_sample_from_segmented_feature(idx, self.segment_length, self.context, self.path_x, self.fn, axis=1)
        if self.path_y is not None:
            y, _ = load_sample_from_segmented_feature(idx, self.segment_length, self.context, self.path_y, fn_y, axis=1)
            if y.shape[0] == 128:
                y = y[min_pitch:min_pitch + n_bins_out, :]
            
        # Get the context by the idx
        x_context = torch.from_numpy(x[:, idx_in_segment - self.context // 2 : idx_in_segment + self.context // 2 + 1, :])
        x_context = x_context.permute(2, 1, 0)   # (n_chan, n_frames, n_bins)
        if self.path_y is not None:
            y_context = torch.from_numpy(y[:, idx_in_segment])
            y_context = y_context.unsqueeze(0).unsqueeze(1)   # (1, 1, n_bins)
        else:
            y_context = torch.zeros(1, 1, n_bins_out, dtype=torch.float32)

        # Data augmentation
        if not self.eval:
            x_context, y_context = data_augmentation(x_context, y_context, self.augmentation)

        if self.task == 'domain_adaptation':
            # Reconstruction
            x_context_recon = x_context.clone()

            # Masks for the MPE backend and the reconstruction
            if self.predict:
                mask_mpe = 1  # Test with MPE
                mask_recon = 0  # Ignore the reconstruction
            else:
                mask_mpe = 0  # Ignore the MPE loss
                mask_recon = 1  # Train with reconstruction

            return x_context, y_context, x_context_recon, mask_mpe, mask_recon
        
        elif self.task == 'teacher_student':
            return x_context, y_context
    
        else:
            raise ValueError("Task not supported.")
