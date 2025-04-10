import os, sys
sys.path.insert(0, os.path.join(sys.path[0], '../../'))
import numpy as np, os, scipy, scipy.spatial, matplotlib.pyplot as plt, IPython.display as ipd
from numba import jit
import librosa
import libfmp.b, libfmp.c3, libfmp.c5
import pandas as pd, pickle, re
from numba import jit
import torch.utils.data
import torch.nn as nn
import libdl.data_preprocessing
from libdl.data_loaders import dataset_context
from libdl.nn_models import basic_cnn_segm_sigmoid, deep_cnn_segm_sigmoid, simple_u_net_largekernels
from libdl.nn_models import simple_u_net_doubleselfattn, simple_u_net_doubleselfattn_twolayers
from libdl.nn_models import u_net_blstm_varlayers, simple_u_net_polyphony_classif_softmax
from libdl.data_preprocessing import compute_hopsize_cqt, compute_hcqt, compute_efficient_hcqt, compute_annotation_array_nooverlap
from torchinfo import summary


dataset = 'Wagner_Ring_allperf'

######################################################################
# Set train, val, test songs and versions
######################################################################

if dataset == 'Schubert_Winterreise':
    train_songs = ['D911-01', 'D911-02', 'D911-03', 'D911-04', 'D911-05', 'D911-06', 'D911-07', 'D911-08', 'D911-09', 'D911-10', 'D911-11', 'D911-12', 'D911-13', ]
    val_songs = ['D911-14', 'D911-15', 'D911-16', ]
    test_songs = ['D911-17', 'D911-18', 'D911-19', 'D911-20', 'D911-21', 'D911-22', 'D911-23', 'D911-24']
    train_versions = ['AL98', 'FI55', 'FI80', 'OL06', 'QU98']
    val_versions = ['FI66', 'TR99']
    test_versions = ['HU33', 'SC06']

elif dataset == 'Wagner_Ring_allperf':
    train_songs = ['WWV086B-2', 'WWV086C-2', 'WWV086D-1', 'WWV086B-3', 'WWV086D-3', 'WWV086C-1', 'WWV086D-2']
    val_songs = ['WWV086C-3', 'WWV086B-1']
    test_songs = ['WWV086A', 'WWV086D-0']
    train_versions = ['MEMBRAN2013', 'DG2012', 'PHILIPS2006', 'EMI2012', 'DECCA2012', 'DG2013', 'DECCA2008', 'OEHMS2013', 'NAXOS2003']
    val_versions = ['PROFIL2013', 'SONY2012', 'MEMBRAN1995']
    test_versions = ['ZYX2012', 'EMI2011', 'ORFEO2010']


######################################################################
# Get teacher model 
######################################################################


# dir_models = os.path.join('/mnt/d/workspace/cross-version-mpe/experiments/MAESTRO/models')
dir_models = os.path.join('/mnt/d/workspace/cross-version-mpe/experiments/Wagner_Ring_allperf/models')
num_octaves_inp = 6
num_output_bins, min_pitch = 72, 24


# Polyphony U-Net trained in recommended MusicNet split (test set MuN-10full):
# model_params = {'n_chan_input': 6,
#                 'n_chan_layers': [128,180,150,100],
#                 'n_ch_out': 2,
#                 'n_bins_in': num_octaves_inp*12*3,
#                 'n_bins_out': num_output_bins,
#                 'a_lrelu': 0.3,
#                 'p_dropout': 0.2,
#                 'scalefac': 2,
#                 'num_polyphony_steps': 24
#                 }
model_params = {'n_chan_input': 6,
                'n_chan_layers': [70,70,50,10],
                'n_prefilt_layers': 5,
                'residual': True,
                'n_bins_in': num_octaves_inp*12*3,
                'n_bins_out': num_output_bins,
                'a_lrelu': 0.3,
                'p_dropout': 0.2
                }
mp = model_params

# fn_model = 'teacher_maestro.pt'
fn_model = 'teacher_student_cross_version_2.v0.6.2.pt'
model = deep_cnn_segm_sigmoid(n_chan_input=mp['n_chan_input'], n_chan_layers=mp['n_chan_layers'], n_prefilt_layers=mp['n_prefilt_layers'], \
    residual=mp['residual'], n_bins_in=mp['n_bins_in'], n_bins_out=mp['n_bins_out'], a_lrelu=mp['a_lrelu'], p_dropout=mp['p_dropout'])
# model = simple_u_net_polyphony_classif_softmax(n_chan_input=mp['n_chan_input'], n_chan_layers=mp['n_chan_layers'], \
#     n_bins_in=mp['n_bins_in'], n_bins_out=mp['n_bins_out'], a_lrelu=mp['a_lrelu'], p_dropout=mp['p_dropout'], \
#     scalefac=mp['scalefac'], num_polyphony_steps=mp['num_polyphony_steps'])


path_trained_model = os.path.join(dir_models, fn_model)

model.load_state_dict(torch.load(path_trained_model, map_location=torch.device('cuda:0')))

model.eval()
summary(model, input_size=(1, 6, 174, 216))


######################################################################
# Path to features and test parameters
######################################################################


# Path to features
hcqt_path = os.path.join('/mnt/d/Datasets/precomputed_features', dataset, 'hcqt_hs512_o6_h5_s1')
# teacher_annotations_path = os.path.join('/mnt/d/workspace/cross-version-mpe/feature_preparation', dataset, 'teacher_annotations')
teacher_annotations_path = os.path.join('/mnt/d/workspace/cross-version-mpe/experiments/Wagner_Ring_allperf/predictions/teacher_student_cross_version_2.v0.6.2')
if not os.path.exists(teacher_annotations_path):
    os.makedirs(teacher_annotations_path)

# Set test parameters
test_params = {'batch_size': 50,
            'shuffle': False,
            'num_workers': 16,
            }
device = 'cuda:0'

test_dataset_params = {'context': 75,
                    # 'stride': 1,
                    'stride': 100,
                    'compression': 10
                    }
half_context = test_dataset_params['context']//2

######################################################################
# Calculate teacher annotations
######################################################################


for i, fn in enumerate(os.listdir(hcqt_path)):
    print('Processing file %d of %d' % (i+1, len(os.listdir(hcqt_path))))

    in_cross_version_set = False

    # if any(version in fn for version in train_versions) and \
    #         any(song in fn for song in train_songs):
    #     in_cross_version_set = True
    # elif any(version in fn for version in val_versions) and \
    #         any(song in fn for song in val_songs):
    #     in_cross_version_set = True
    if any(version in fn for version in test_versions) and \
            any(song in fn for song in test_songs):
        in_cross_version_set = True

    if in_cross_version_set:

        f_hcqt = np.load(os.path.join(hcqt_path, fn))
        print('HCQT shape:', f_hcqt.shape)

        inputs = np.transpose(f_hcqt, (2, 1, 0))
        targets = np.zeros(inputs.shape[1:]) # need dummy targets to use dataset object

        inputs_context = torch.from_numpy(np.pad(inputs, ((0, 0), (half_context, half_context+1), (0, 0))))
        targets_context = torch.from_numpy(np.pad(targets, ((half_context, half_context+1), (0, 0))))

        test_set = dataset_context(inputs_context, targets_context, test_dataset_params)
        test_generator = torch.utils.data.DataLoader(test_set, **test_params)

        pred_tot = np.zeros((0, num_output_bins))

        # max_frames = 160
        # k=0
        for j, (test_batch, test_labels) in enumerate(test_generator):
            print('Processing batch %d of %d' % (j+1, len(test_generator)), end='\r')
            # k+=1
            # if k>max_frames:
            #     break
            # Model computations
            # y_pred, n_pred = model(test_batch.to(device))
            y_pred = model(test_batch.to(device))
            # print(y_pred.shape)
            # input()
            pred_log = torch.squeeze(torch.squeeze(y_pred.to('cpu'),2),1).detach().numpy()
            # pred_log = torch.squeeze(y_pred.to('cuda:0')).detach().numpy()
            pred_tot = np.append(pred_tot, pred_log, axis=0)
            
        predictions = pred_tot
        
        print()
        print('prediction shape:', predictions.shape)

        # Save teacher annotations
        fn_teacher = os.path.join(teacher_annotations_path, fn[:-4])
        np.save(fn_teacher, predictions)



