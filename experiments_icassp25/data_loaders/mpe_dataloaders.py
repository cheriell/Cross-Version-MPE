import torch
import json
import os
import numpy as np
import pandas as pd

from experiments.feature_preparation.utils import (
    get_feature_segment_fn,
    get_feature_segment_metainfo_fn,
    get_version_pairs_fn,
    extend_fn,
)


def get_mpe_dataloader(fns, path_x, path_y, dataset_params, dataloader_params, path_valid_frames=None, eval=False, pick_pairs=False, version_pairs=None):

    all_datasets = []
    for fn in fns:
        dataset = MPEDataset(fn, path_x, path_y, dataset_params, path_valid_frames, eval, pick_pairs, version_pairs)
        all_datasets.append(dataset)
    
    dataset = torch.utils.data.ConcatDataset(all_datasets)
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_params)

    return dataloader

class MPEDataset(torch.utils.data.Dataset):

    def __init__(self, fn, path_x, path_y, dataset_params, path_valid_frames=None, eval=False, pick_pairs=False, version_pairs=None):
        # valid_frames is a list of indices to be used for cross-version training (e.g. consistent indices bwteen versions)
        self.fn = fn
        self.path_x = path_x
        self.path_y = path_y
        self.path_valid_frames = path_valid_frames
        self.eval = eval
        self.pick_pairs = pick_pairs
        
        self.context = dataset_params['context']
        self.stride = dataset_params['stride']
        self.compression = dataset_params['compression']
        
        # Data augmentation
        if not eval:
            self.transposition = dataset_params['aug:transpsemitones']
            self.randomeq = dataset_params['aug:randomeq']
            self.noisestd = dataset_params['aug:noisestd']
            self.tuning = dataset_params['aug:tuning']

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
                idx = valid_frames[index]
            else:
                idx = np.random.choice(valid_frames)
        else:
            idx = index * self.stride + self.context // 2

        # Load inputs and targets
        segment_idx = idx // (self.segment_length * 100)
        idx_in_segment = idx % (self.segment_length * 100)
        x = np.load(os.path.join(self.path_x, get_feature_segment_fn(self.fn, segment_idx)))  # (n_bins, n_frames, n_chan)
        y = np.load(os.path.join(self.path_y, get_feature_segment_fn(fn_y, segment_idx)))   # (n_bins, n_frames)

        if idx_in_segment < self.context // 2:
            idx_in_segment += self.segment_length * 100
            segment_idx -= 1
            x_prev = np.load(os.path.join(self.path_x, get_feature_segment_fn(self.fn, segment_idx)))
            y_prev = np.load(os.path.join(self.path_y, get_feature_segment_fn(fn_y, segment_idx)))
            x = np.concatenate((x_prev, x), axis=1)
            y = np.concatenate((y_prev, y), axis=1)

        elif idx_in_segment > self.segment_length * 100 - self.context // 2 - 1:
            x_next = np.load(os.path.join(self.path_x, get_feature_segment_fn(self.fn, segment_idx + 1)))
            y_next = np.load(os.path.join(self.path_y, get_feature_segment_fn(fn_y, segment_idx + 1)))
            # pad x_next and y_next if the length (axis=1) is shorter than half of the context
            if x_next.shape[1] < self.context // 2:
                x_next = np.concatenate((x_next, np.zeros((x_next.shape[0], self.context // 2 - x_next.shape[1], x_next.shape[2]))), axis=1)
                y_next = np.concatenate((y_next, np.zeros((y_next.shape[0], self.context // 2 - y_next.shape[1]))), axis=1)
            x = np.concatenate((x, x_next), axis=1)
            y = np.concatenate((y, y_next), axis=1)

        # Get the context by the idx
        x_context = torch.from_numpy(x[:, idx_in_segment - self.context // 2 : idx_in_segment + self.context // 2 + 1, :])
        y_context = torch.from_numpy(y[:, idx_in_segment])
        # transpose
        x_context = x_context.permute(2, 1, 0)   # (n_chan, n_frames, n_bins)
        y_context = y_context.unsqueeze(0).unsqueeze(1)   # (1, 1, n_bins)

        # Data augmentation
        if not self.eval:
            x_context, y_context = self.data_augmentation(x_context, y_context)

        return x_context, y_context


    def data_augmentation(self, X, y):
        
        if self.transposition:
            transp = torch.randint(-self.transposition, self.transposition+1, (1, ))
            X_trans = torch.roll(X, (transp.item()*3, ), -1)
            y_trans = torch.roll(y, (transp.item(), ), -1)
            if transp>0:
                X_trans[:, :, :(3*transp)] = torch.abs(torch.normal(mean=torch.zeros(X_trans[:, :, :(3*transp)].size()), std=1e-4*torch.ones(X_trans[:, :, :(3*transp)].size())))
                y_trans[:, :, :transp] = torch.zeros(y_trans[:, :, :transp].size())
            elif transp<0:
                X_trans[:, :, (3*transp):] = torch.abs(torch.normal(mean=torch.zeros(X_trans[:, :, (3*transp):].size()), std=1e-4*torch.ones(X_trans[:, :, (3*transp):].size())))
                y_trans[:, :, transp:] = torch.zeros(y_trans[:, :, transp:].size())
            if y_trans.size(-1)==12:
                y_trans = torch.roll(y, (transp.item(), ), -1)
            X = X_trans
            y = y_trans

        if self.randomeq:
            minval = -1
            while minval<0:
                randomAlpha = torch.randint(1, self.randomeq+1, (1,))
                randomBeta = torch.randint(0, 216, (1,))
                # filtvec = ((1 - (2e-6*randomAlpha*(torch.arange(216)-randomBeta)**2)).unsqueeze(0).unsqueeze(0))
                filtmat = torch.zeros((X.size(0), 1, X.size(2)))
                for nharm in range(filtmat.size(0)):
                    if nharm==0:
                        offset = int(-3*12)
                    else:
                        offset = int(3*12*(np.log2(nharm)))
                    randomBetaHarm = randomBeta - offset
                    currfiltvec = ((1 - (2e-6*randomAlpha*(torch.arange(216)-randomBetaHarm)**2)).unsqueeze(0).unsqueeze(0))
                    filtmat[nharm, :, :] = currfiltvec
                minval = torch.min(filtmat)
            X_filt = filtmat*X
            X = X_filt

        if self.noisestd:
            X += torch.normal(mean=torch.zeros(X.size()), std=self.noisestd*torch.ones(X.size()))
            X_noise = torch.abs(X)
            X = X_noise

        if self.tuning:
            tuneshift = torch.randint(-2, 3, (1, )).item()
            tuneshift /= 2.
            X_tuned = X
            if tuneshift==0.5:
                # +0.5:
                X_tuned[:, :, 1:] = (X[:, :, :-1] + X[:, :, 1:])/2
            elif tuneshift==-0.5:
                # -0.5
                X_tuned[:, :, :-1] = (X[:, :, :-1] + X[:, :, 1:])/2
            else:
                X_tuned = torch.roll(X, (int(tuneshift), ), -1)
            if tuneshift>0:
                X_tuned[:, :, :1] = torch.abs(torch.normal(mean=torch.zeros(X_tuned[:, :, :1].size()), std=1e-4*torch.ones(X_tuned[:, :, :1].size())))
            elif tuneshift<0:
                X_tuned[:, :, -1:] = torch.abs(torch.normal(mean=torch.zeros(X_tuned[:, :, -1:].size()), std=1e-4*torch.ones(X_tuned[:, :, -1:].size())))
            X = X_tuned

        return X, y