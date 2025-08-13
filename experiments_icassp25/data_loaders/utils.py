import numpy as np
import os
import torch

from experiments_icassp25.feature_preparation.utils import (
    get_feature_segment_fn
)


###########################################################
# Load sample from segmented feature
###########################################################


def load_sample_from_segmented_feature(idx, segment_length, context, path_feature, fn_feature, axis=1):
    segment_idx = idx // (segment_length * 100)
    idx_in_segment = idx % (segment_length * 100)
    x = np.load(os.path.join(path_feature, get_feature_segment_fn(fn_feature, segment_idx)))  # (n_bins, n_frames, n_chan), or (n_bins, n_frames)

    if idx_in_segment < context // 2:
        idx_in_segment += segment_length * 100
        segment_idx -= 1
        x_prev = np.load(os.path.join(path_feature, get_feature_segment_fn(fn_feature, segment_idx)))
        x = np.concatenate((x_prev, x), axis=axis)

    elif idx_in_segment > segment_length * 100 - context // 2 - 1:
        x_next = np.load(os.path.join(path_feature, get_feature_segment_fn(fn_feature, segment_idx + 1)))
        # pad x_next and y_next if the length (axis=1) is shorter than half of the context
        if x_next.shape[1] < context // 2:
            x_pad_shape = list(x_next.shape); x_pad_shape[axis] = context // 2 - x_next.shape[axis]
            x_next = np.concatenate((x_next, np.zeros(x_pad_shape)), axis=axis)
        x = np.concatenate((x, x_next), axis=axis)

    return x, idx_in_segment


###########################################################
# Data augmentation
###########################################################

def data_augmentation(X, y, augmentation):

    transposition = augmentation['transpsemitones']
    randomeq = augmentation['randomeq']
    noisestd = augmentation['noisestd']
    tuning = augmentation['tuning']
    
    if transposition:
        transp = torch.randint(-transposition, transposition+1, (1, ))
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

    if randomeq:
        minval = -1
        while minval<0:
            randomAlpha = torch.randint(1, randomeq+1, (1,))
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

    if noisestd:
        X += torch.normal(mean=torch.zeros(X.size()), std=noisestd*torch.ones(X.size()))
        X_noise = torch.abs(X)
        X = X_noise

    if tuning:
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


###########################################################
# Self-defined sampler (half from source and half from target per batch)
###########################################################

class HalfHalfSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, source_dataset_length, target_dataset_length, shuffle=True):

        self.source_dataset_length = source_dataset_length
        self.target_dataset_length = target_dataset_length
        self.shuffle = shuffle

        self.len = np.max([self.source_dataset_length, self.target_dataset_length]) * 2
        self.len = np.min([self.len, 50 * 5000])  # limit the maximum length to 50 * 5000, to avoid too long epoch.

    def __iter__(self):
        # within each batch, half is from source and half is from target
        # source: idx = 0, 2, 4, 6, 8, ...
        # target: idx = 1, 3, 5, 7, 9, ...

        if self.shuffle:
            self.source_indices = torch.randperm(self.source_dataset_length).tolist()
            self.target_indices = (torch.randperm(self.target_dataset_length) + self.source_dataset_length).tolist()
        else:
            self.source_indices = list(np.arange(self.source_dataset_length))
            self.target_indices = list(np.arange(self.target_dataset_length) + self.source_dataset_length)

        batch_indices = []
        for i in range(self.len // 2):
            batch_indices.append(self.source_indices[i % len(self.source_indices)])
            batch_indices.append(self.target_indices[i % len(self.target_indices)])

        return iter(batch_indices)
    
    def __len__(self):
        return self.len