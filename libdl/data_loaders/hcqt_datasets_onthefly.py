import numpy as np
from scipy import signal
import torch
import torch.utils.data
import torch.nn as nn
from torchvision import transforms
from scipy.interpolate import interp1d
import os
from multiprocessing import Pool



class dataset_context(torch.utils.data.Dataset):
    """
    Dataset class to be used with DataLoader object. Generates a single HCQT
    frame with context. Note that X (HCQT input) includes the context frames
    but y (pitch (class) target) only refers to the center frame to be predicted.

    Args:
        inputs:         Tensor of HCQT input for one audio file
        targets:        Tensor of pitch (class) targets for the same audio file
        parameters:     Dictionary of parameters with:
        - 'context':        Total number of frames including context frames
        - 'stride':         Hopsize for jumping to the start frame of the next segment
        - 'compression':    Gamma parameter for log compression of HCQT input
        - 'targettype':     'pitch_class' (assumed) or 'pitch'
        - 'aug:transpsemitones':  Data augmentation with transposition (# semitones)
        - 'aug:scalingfactor':    Data augmentation with time scaling (factor)
        - 'aug:randomeq':         Data augmentation with random frequency equalization (amount)
        - 'aug:noisestd':         Data augmentation with random Gaussian noise (standard dev.)
        - 'aug:tuning':           Data augmentation with random tuning shift (+/- 1/3 semitone)
    """
    def __init__(self, inputs, targets, params, fn, agreed_index=None, eval=False, use_buffer=True):
        # Initialization
        torch.initial_seed()
        # self.inputs = inputs
        # self.targets = targets


        self.context = params['context']
        self.stride = params['stride']
        self.compression = params['compression']
        if 'targettype' not in params:
            params['targettype'] = 'pitch_class'
        self.targettype = params['targettype']
        self.transposition = None
        self.scalingfactor = None
        self.randomeq = None
        self.noisestd = None
        self.tuning = None
        if 'aug:transpsemitones' in params:
            self.transposition = params['aug:transpsemitones']
        if 'aug:scalingfactor' in params:
            self.scalingfactor = params['aug:scalingfactor']
        if 'aug:randomeq' in params:
            self.randomeq = params['aug:randomeq']
        if 'aug:noisestd' in params:
            self.noisestd = params['aug:noisestd']
        if 'aug:tuning' in params:
            self.tuning = params['aug:tuning']
        if 'aug:smooth_len' in params and params['aug:smooth_len']>1:
            filt_kernel = np.expand_dims(signal.get_window(params['aug:smooth_win'], params['aug:smooth_len']+1)[1:], axis=1)
            targets = signal.convolve(targets, filt_kernel, mode='same')
            targets /= np.max(targets)
            targets = torch.from_numpy(targets)

        # Save inputs and targets into files
        self.fn = fn
        self.use_buffer = use_buffer
        if use_buffer:
            np.save(fn+'_inputs.npy', inputs.numpy())
            np.save(fn+'_targets.npy', targets.numpy())
        else:
            self.inputs = inputs
            self.targets = targets

        # Get length of the dataset
        if agreed_index is None:
            self.len = (inputs.size()[1]-self.context)//self.stride
            self.use_agreed_index = False
        else:
            # force the agreed index to be within the half context boundaries
            agreed_index = agreed_index[agreed_index>=self.context//2]
            agreed_index = agreed_index[agreed_index<(inputs.size()[1]-self.context//2)]

            # Force the length to be the same as if we didn't use agreed index
            if eval:  # if eval, use all agreed index
                self.len = agreed_index.size
            else:
                self.len = (inputs.size()[1]-self.context)//self.stride

            # Save agreed index
            np.save(fn+'_agreed_index.npy', agreed_index)
            self.use_agreed_index = True

        self.eval = eval

        print('Dataset length: ', self.len)

    def __len__(self):
        # Denotes the total number of samples
        return self.len

    def __getitem__(self, index):
        # Generates one sample of data

        # Load inputs and targets
        if self.use_buffer:
            inputs = torch.from_numpy(np.load(self.fn+'_inputs.npy'))
            targets = torch.from_numpy(np.load(self.fn+'_targets.npy'))
        else:
            inputs = self.inputs
            targets = self.targets

        half_context = self.context//2

        if self.use_agreed_index:
            agreed_index = np.load(self.fn+'_agreed_index.npy')
            if self.eval:
                index = agreed_index[index]
            else:
                # Randomly select index
                index = np.random.choice(agreed_index)
        else:
            # shift index by half context
            index *= self.stride
            index += half_context
        # Load data and get label (remove subharmonic)
        X = inputs[:, (index-half_context):(index+half_context+1), :].type(torch.FloatTensor)
        y = torch.unsqueeze(torch.unsqueeze(targets[index, :], 0), 1).type(torch.FloatTensor)

        if self.scalingfactor:
            assert False, 'Scaling not implemented for dataset_context!'

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
            # X_pos = (X>0).type('torch.FloatTensor')

        if self.compression is not None:
            X = np.log(1+self.compression*X)

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

        return X, y
    

    
class dataset_context_fix_length_and_buffer_segmentwise(torch.utils.data.Dataset):
    """
    Dataset class to be used with DataLoader object. Generates a single HCQT
    frame with context. Note that X (HCQT input) includes the context frames
    but y (pitch (class) target) only refers to the center frame to be predicted.

    Args:
        inputs:         Tensor of HCQT input for one audio file
        targets:        Tensor of pitch (class) targets for the same audio file
        parameters:     Dictionary of parameters with:
        - 'context':        Total number of frames including context frames
        - 'stride':         Hopsize for jumping to the start frame of the next segment
        - 'compression':    Gamma parameter for log compression of HCQT input
        - 'targettype':     'pitch_class' (assumed) or 'pitch'
        - 'aug:transpsemitones':  Data augmentation with transposition (# semitones)
        - 'aug:scalingfactor':    Data augmentation with time scaling (factor)
        - 'aug:randomeq':         Data augmentation with random frequency equalization (amount)
        - 'aug:noisestd':         Data augmentation with random Gaussian noise (standard dev.)
        - 'aug:tuning':           Data augmentation with random tuning shift (+/- 1/3 semitone)
    """
    def __init__(self, inputs, targets, params, fn, agreed_index=None, eval=False, inputs_len=None):
        # Initialization
        torch.initial_seed()
        # self.inputs = inputs
        # self.targets = targets


        self.context = params['context']
        self.stride = params['stride']
        self.compression = params['compression']
        if 'targettype' not in params:
            params['targettype'] = 'pitch_class'
        self.targettype = params['targettype']
        self.transposition = None
        self.scalingfactor = None
        self.randomeq = None
        self.noisestd = None
        self.tuning = None
        if 'aug:transpsemitones' in params:
            self.transposition = params['aug:transpsemitones']
        if 'aug:scalingfactor' in params:
            self.scalingfactor = params['aug:scalingfactor']
        if 'aug:randomeq' in params:
            self.randomeq = params['aug:randomeq']
        if 'aug:noisestd' in params:
            self.noisestd = params['aug:noisestd']
        if 'aug:tuning' in params:
            self.tuning = params['aug:tuning']
        # if 'aug:smooth_len' in params and params['aug:smooth_len']>1:
        #     filt_kernel = np.expand_dims(signal.get_window(params['aug:smooth_win'], params['aug:smooth_len']+1)[1:], axis=1)
        #     targets = signal.convolve(targets, filt_kernel, mode='same')
        #     targets /= np.max(targets)
        #     targets = torch.from_numpy(targets)

        # Save inputs and targets into files
        self.fn_X, self.fn_y, self.fn_other = fn
        
        # # Buffer segmentwise (segment length: 5000 frames)
        # print(inputs.shape, targets.shape)

        # for seg_idx in range(np.ceil(inputs.shape[1] / 5000).astype(int)):
        #     print('Saving segment %d of %d' % (seg_idx+1, np.ceil(inputs.shape[1] / 5000).astype(int)), end='\r')
        #     if not os.path.exists(fn+'_inputs_seg_{}.npy'.format(seg_idx)):
        #         np.save(fn+'_inputs_seg_{}.npy'.format(seg_idx), inputs[:, seg_idx*5000:(seg_idx+1)*5000, :].numpy())
        #     if not os.path.exists(fn+'_targets_seg_{}.npy'.format(seg_idx)):
        #         np.save(fn+'_targets_seg_{}.npy'.format(seg_idx), targets[seg_idx*5000:(seg_idx+1)*5000, :].numpy())
        
        # print('')
        # np.save(fn+'_inputs.npy', inputs.numpy())
        # np.save(fn+'_targets.npy', targets.numpy())

        # Get length of the dataset
        if inputs_len is None:
            inputs_len = inputs.shape[1]
            
        if agreed_index is None:
            self.len = (inputs_len-self.context)//self.stride
            self.use_agreed_index = False
        else:
            # force the agreed index to be within the half context boundaries
            agreed_index = agreed_index[agreed_index>=self.context//2]
            agreed_index = agreed_index[agreed_index<(inputs_len-self.context//2)]

            # Force the length to be the same as if we didn't use agreed index
            if eval:  # if eval, use all agreed index
                agreed_index = agreed_index[::100]
                self.len = len(agreed_index)
                # self.len = min(300, self.len)
            else:
                self.len = (inputs_len-self.context)//self.stride
                self.len = self.len // 16  # update this to be shorter and in a.c. parallel with the Schubert dataset.

            # Save agreed index
            np.save(self.fn_other+'_agreed_index.npy', agreed_index)
            self.use_agreed_index = True

        self.eval = eval

        print('Dataset length: ', self.len)

    def __len__(self):
        # Denotes the total number of samples
        return self.len

    def __getitem__(self, index):
        # Generates one sample of data


        # inputs = torch.from_numpy(np.load(self.fn+'_inputs.npy'))
        # targets = torch.from_numpy(np.load(self.fn+'_targets.npy'))

        half_context = self.context//2

        if self.use_agreed_index:
            agreed_index = np.load(self.fn_other+'_agreed_index.npy')
            if self.eval:
                index = agreed_index[index]
            else:
                # Randomly select index
                index = np.random.choice(agreed_index)
        else:
            # shift index by half context
            index *= self.stride
            index += half_context
        
        # # Load inputs and targets
        # Get seg_idx
        seg_idx = (index-half_context) // 5000
        # Load inputs and targets
        inputs = torch.from_numpy(np.load(self.fn_X+'_inputs_seg_{}.npy'.format(seg_idx)))
        targets = torch.from_numpy(np.load(self.fn_y+'_targets_seg_{}.npy'.format(seg_idx)))
        if (index+half_context) // 5000 > seg_idx:
            inputs = torch.cat((inputs, torch.from_numpy(np.load(self.fn_X+'_inputs_seg_{}.npy'.format(seg_idx+1)))), dim=1)
            targets = torch.cat((targets, torch.from_numpy(np.load(self.fn_y+'_targets_seg_{}.npy'.format(seg_idx+1)))), dim=0)

        # Update index to the index within the segment
        index = index - seg_idx*5000

        # Load data and get label (remove subharmonic)
        X = inputs[:, (index-half_context):(index+half_context+1), :].type(torch.FloatTensor)
        y = torch.unsqueeze(torch.unsqueeze(targets[index, :], 0), 1).type(torch.FloatTensor)

        if self.scalingfactor:
            assert False, 'Scaling not implemented for dataset_context!'

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
            # X_pos = (X>0).type('torch.FloatTensor')

        if self.compression is not None:
            X = np.log(1+self.compression*X)

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

        return X, y