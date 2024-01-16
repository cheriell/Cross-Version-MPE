import os
import sys
sys.path.insert(0, os.path.join(sys.path[0], '../../'))
import numpy as np, os, scipy, scipy.spatial, matplotlib.pyplot as plt, IPython.display as ipd
from itertools import groupby
from numba import jit
import librosa
import libfmp.c3, libfmp.c5
import pandas as pd, pickle, re
from numba import jit
import torch
import torch.utils.data
import torch.nn as nn
from torchinfo import summary
from libdl.data_loaders import dataset_context
from libdl.data_loaders.hcqt_datasets_onthefly import dataset_context_fix_length_and_buffer_segmentwise
from libdl.nn_models import deep_cnn_segm_sigmoid
from libdl.metrics import early_stopping, calculate_eval_measures, calculate_mpe_measures_mireval
import logging
from multiprocessing import Pool


################################################################################
#### Set experimental configuration ############################################
################################################################################
version_str = 'v0.6.2-repeat2'
agreed_frames_folder = 'two_version_agreed_frames.eval_thresh=0.50.pitched_only=True.allow_diff=2.align_tolerance=2'
dataset = 'Wagner_Ring_allperf'

resume_from_checkpoint_path = '/home/lel79bc/workspace/cross-version-mpe/experiments/Wagner_Ring_allperf/models/teacher_student_cross_version_2.v0.6.2-repeat1.pt'
resume_training = False

save_dataset_buffer = False


# Get experiment name from script name
curr_filepath = sys.argv[0]
expname = curr_filename = os.path.splitext(os.path.basename(curr_filepath))[0]
expname = '.'.join([expname, version_str])
print(' ... running experiment ' + expname)

# Which steps to perform
do_train = True
do_val = True
do_test = True
store_results_filewise = True
store_predictions = True

# Set training parameters
train_dataset_params = {'context': 75,
                        'stride': 50,
                        'compression': 10,
                        'aug:transpsemitones': 5,
                        'aug:randomeq': 20,
                        'aug:noisestd': 1e-4,
                        'aug:tuning': True
                        }
val_dataset_params = {'context': 75,
                    #   'stride': 50,
                      'stride': 5000,  # change to a more sparse stride for validation
                      'compression': 10
                      }
test_dataset_params = {'context': 75,
                    #    'stride': 1,
                       'stride': 100, # change to a more sparse stride for testing
                       'compression': 10
                      }
train_params = {'batch_size': 25,
                'shuffle': True,
                'num_workers': 16,
                }
val_params = {'batch_size': 50,
              'shuffle': False,
              'num_workers': 16
              }
test_params = {'batch_size': 50,
              'shuffle': False,
              'num_workers': 8
              }


# Specify model ################################################################

num_octaves_inp = 6
num_output_bins, min_pitch = 72, 24
# num_output_bins = 12
model_params = {'n_chan_input': 6,
                'n_chan_layers': [70,70,50,10],
                'n_prefilt_layers': 5,
                'residual': True,
                'n_bins_in': num_octaves_inp*12*3,
                'n_bins_out': num_output_bins,
                'a_lrelu': 0.3,
                'p_dropout': 0.2
                }

if do_train:

    # max_epochs = 100
    max_epochs = 50  # smaller max_epochs

# Specify criterion (loss) #####################################################
    criterion = torch.nn.BCELoss(reduction='mean')
    # criterion = sctc_loss_threecomp()
    # criterion = sctc_loss_twocomp()
    # criterion = mctc_ne_loss_twocomp()
    # criterion = mctc_ne_loss_threecomp()
    # criterion = mctc_we_loss()
    # criterion = torch.nn.MSELoss(reduction='mean')  # use MSE loss for student trainings -> this is not working well, resume to binary crossentropy



# Set optimizer and parameters #################################################
    # optimizer_params = {'name': 'SGD',
    #                     'initial_lr': 0.01,
    #                     'momentum': 0.9}
    # optimizer_params = {'name': 'Adam',
    #                     'initial_lr': 0.01,
    #                     'betas': [0.9, 0.999]}
    optimizer_params = {'name': 'AdamW',
                        'initial_lr': 0.0002,
                        'betas': (0.9, 0.999),
                        'eps': 1e-08,
                        'weight_decay': 0.01,
                        'amsgrad': False}


# Set scheduler and parameters #################################################
    # scheduler_params = {'use_scheduler': True,
    #                     'name': 'LambdaLR',
    #                     'start_lr': 1,
    #                     'end_lr': 1e-2,
    #                     'n_decay': 20,
    #                     'exp_decay': .5
    #                     }
    scheduler_params = {'use_scheduler': True,
                        'name': 'ReduceLROnPlateau',
                        'mode': 'min',
                        'factor': 0.5,
                        'patience': 5,
                        'threshold': 0.0001,
                        'threshold_mode': 'rel',
                        'cooldown': 0,
                        'min_lr': 1e-6,
                        'eps': 1e-08,
                        'verbose': False
                        }


# Set early_stopping and parameters ############################################
    early_stopping_params = {'use_early_stopping': True,
                             'mode': 'min',
                             'min_delta': 1e-5,
                             'patience': 12,
                             'percentage': False
                             }


# Set evaluation measures to compute while testing #############################
if do_test:
    eval_thresh = 0.4
    eval_measures = ['precision', 'recall', 'f_measure', 'cosine_sim', 'binary_crossentropy', \
            'euclidean_distance', 'binary_accuracy', 'soft_accuracy', 'accum_energy', 'roc_auc_measure', 'average_precision_score']


# Specify paths and splits #####################################################
# path_data_basedir = os.path.join(os.sep, 'tsi', 'clusterhome', 'cweiss', 'data')
path_data_basedir = os.path.join('/mnt/d/Datasets/precomputed_features')
path_data = os.path.join(path_data_basedir, dataset, 'hcqt_hs512_o6_h5_s1')
# path_data = os.path.join(path_data_basedir, 'MusicNet', 'hcqt_hs512_o6_h5_s1')
# path_annot = os.path.join(path_data_basedir, 'MusicNet', 'pitchclass_hs512')
# path_annot = os.path.join(path_data_basedir, 'MusicNet', 'pitchclass_hs512_nooverl')
# path_annot = os.path.join(path_data_basedir, 'MusicNet', 'pitchclass_hs512_shorten75')
# path_annot = os.path.join(path_data_basedir, 'MusicNet', 'pitchclass_hs512_shorten50')
# path_annot = os.path.join(path_data_basedir, 'MusicNet', 'pitchclass_hs512_shorten25')
path_annot_test = os.path.join(path_data_basedir, dataset, 'pitch_hs512_nooverl')
# path_annot = os.path.join(path_data_basedir, 'MusicNet', 'pitch_hs512_nooverl')
# path_annot = os.path.join(path_data_basedir, 'MusicNet', 'pitch_hs512_shorten75')
# path_annot = os.path.join(path_data_basedir, 'MusicNet', 'pitch_hs512_shorten50')
# path_annot = os.path.join(path_data_basedir, 'MusicNet', 'pitch_hs512_shorten25')
# Use teacher annotations as annotations
path_annot = os.path.join('/home/lel79bc/workspace/cross-version-mpe/feature_preparation', dataset, 'teacher_annotations')

# Path to the agreed index for the cross-version teacher annotations
path_agreed_index = os.path.join('/home/lel79bc/workspace/cross-version-mpe/feature_preparation', dataset, agreed_frames_folder)


# Where to save models
# dir_models = os.path.join(os.sep, 'tsi', 'clusterhome', 'cweiss', 'models')
dir_models = os.path.join('/home/lel79bc/workspace/cross-version-mpe/experiments', dataset, 'models')
if not os.path.exists(dir_models):
    os.makedirs(dir_models)
fn_model = expname + '.pt'
path_trained_model = os.path.join(dir_models, fn_model)

# Where to save results
# dir_output = os.path.join(os.sep, 'tsi', 'clusterhome', 'cweiss', 'code', 'deep_pitch_estimation', 'experiments', 'results_filewise')
dir_output = os.path.join('/home/lel79bc/workspace/cross-version-mpe/experiments', dataset, 'results_filewise')
if not os.path.exists(dir_output):
    os.makedirs(dir_output)
fn_output = expname + '.csv'
path_output = os.path.join(dir_output, fn_output)

# Where to save predictions
# dir_predictions = os.path.join(os.sep, 'tsi', 'clusterhome', 'cweiss', 'predictions', expname)
dir_predictions = os.path.join('/home/lel79bc/workspace/cross-version-mpe/experiments', dataset, 'predictions', expname)
if not os.path.exists(dir_predictions):
    os.makedirs(dir_predictions)

# Where to save logs
fn_log = expname + '.txt'
# path_log = os.path.join(os.sep, 'tsi', 'clusterhome', 'cweiss', 'code', 'deep_pitch_estimation', 'experiments', 'logs', fn_log)
path_log = os.path.join('/home/lel79bc/workspace/cross-version-mpe/experiments', dataset, 'logs', fn_log)
if not os.path.exists(os.path.dirname(path_log)):
    os.makedirs(os.path.dirname(path_log))

# Where to save dataset buffer
path_dataset_buffer_X = os.path.join('/home/lel79bc/workspace/cross-version-mpe/experiments', dataset, 'dataset_buffer', 'X')
path_dataset_buffer_y = os.path.join('/home/lel79bc/workspace/cross-version-mpe/experiments', dataset, 'dataset_buffer', 'y_teacher')
path_dataset_buffer_other = os.path.join('/home/lel79bc/workspace/cross-version-mpe/experiments', dataset, 'dataset_buffer', expname)
if not os.path.exists(path_dataset_buffer_X):
    os.makedirs(path_dataset_buffer_X)
if not os.path.exists(path_dataset_buffer_y):
    os.makedirs(path_dataset_buffer_y)
if not os.path.exists(path_dataset_buffer_other):
    os.makedirs(path_dataset_buffer_other)

# Log basic configuration
logging.basicConfig(filename=path_log, filemode='w', format='%(asctime)s | %(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logging.info('Logging experiment ' + expname)
logging.info('Experiment config: do training = ' + str(do_train))
logging.info('Experiment config: do validation = ' + str(do_val))
logging.info('Experiment config: do testing = ' + str(do_test))
logging.info("Training set parameters: {0}".format(train_dataset_params))
logging.info("Validation set parameters: {0}".format(val_dataset_params))
logging.info("Test set parameters: {0}".format(test_dataset_params))
if do_train:
    logging.info("Training parameters: {0}".format(train_params))
    logging.info('Trained model saved in ' + path_trained_model)
# Log criterion, optimizer, and scheduler ######################################
    logging.info(' --- Training config: ----------------------------------------- ')
    logging.info('Maximum number of epochs: ' + str(max_epochs))
    logging.info('Criterion (Loss): ' + criterion.__class__.__name__)
    logging.info("Optimizer parameters: {0}".format(optimizer_params))
    logging.info("Scheduler parameters: {0}".format(scheduler_params))
    logging.info("Early stopping parameters: {0}".format(early_stopping_params))
if do_test:
    logging.info("Test parameters: {0}".format(test_params))
    logging.info('Save filewise results = ' + str(store_results_filewise) + ', in folder ' + path_output)
    logging.info('Save model predictions = ' + str(store_predictions) + ', in folder ' + dir_predictions)


################################################################################
#### Start experiment ##########################################################
################################################################################

# CUDA for PyTorch #############################################################
use_cuda = torch.cuda.is_available()
assert use_cuda, 'No GPU found! Exiting.'
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
logging.info('CUDA use_cuda: ' + str(use_cuda))
logging.info('CUDA device: ' + str(device))

# Specify and log model config #################################################
# Specify and log model config #################################################
mp = model_params
model = deep_cnn_segm_sigmoid(n_chan_input=mp['n_chan_input'], n_chan_layers=mp['n_chan_layers'], n_prefilt_layers=mp['n_prefilt_layers'], \
    residual=mp['residual'], n_bins_in=mp['n_bins_in'], n_bins_out=mp['n_bins_out'], a_lrelu=mp['a_lrelu'], p_dropout=mp['p_dropout'])
model.to(device)

# Resume from pre-trained model
if resume_training:
    model.load_state_dict(torch.load(resume_from_checkpoint_path))

logging.info(' --- Model config: -------------------------------------------- ')
logging.info('Model: ' + model.__class__.__name__)
logging.info("Model parameters: {0}".format(model_params))
logging.info('\n' + str(summary(model, input_size=(1, 6, 174, 216))))

# Generate training dataset ####################################################
if do_val:
    assert do_train, 'Validation without training not possible!'
# train_songs = ['D911-01', 'D911-02', 'D911-03', 'D911-04', 'D911-05', 'D911-06', 'D911-07', 'D911-08', 'D911-09', 'D911-10', 'D911-11', 'D911-12', 'D911-13', ]
# val_songs = ['D911-14', 'D911-15', 'D911-16', ]
# test_songs = ['D911-17', 'D911-18', 'D911-19', 'D911-20', 'D911-21', 'D911-22', 'D911-23', 'D911-24']
# train_versions = ['AL98', 'FI55', 'FI80', 'OL06', 'QU98']
# val_versions = ['FI66', 'TR99']
# test_versions = ['HU33', 'SC06']
# test_versions_small = test_versions   # Set it to be the same as test_versions

# Change to neither split in the Wagner Ring dataset
train_songs = ['WWV086B-2', 'WWV086C-2', 'WWV086D-1', 'WWV086B-3', 'WWV086D-3', 'WWV086C-1', 'WWV086D-2']
val_songs = ['WWV086C-3', 'WWV086B-1']
test_songs = ['WWV086A', 'WWV086D-0']
train_versions = ['MEMBRAN2013', 'DG2012', 'PHILIPS2006', 'EMI2012', 'DECCA2012', 'DG2013', 'DECCA2008', 'OEHMS2013', 'NAXOS2003']
val_versions = ['PROFIL2013', 'SONY2012', 'MEMBRAN1995']
test_versions = ['ZYX2012', 'EMI2011', 'ORFEO2010']
test_versions_small = test_versions   # Set it to be the same as test_versions

all_train_fn = []
all_train_sets = []
all_val_fn = []
all_val_sets = []

if do_train:
    for fn in os.listdir(path_data):
        if any(train_version in fn for train_version in train_versions) and \
            any(train_song in fn for train_song in train_songs):
            all_train_fn.append(fn)
            if save_dataset_buffer:
                inputs = np.transpose(np.load(os.path.join(path_data, fn)), (2, 1, 0))
            # targets = torch.from_numpy(np.load(os.path.join(path_annot, fn)).T)
            # if num_output_bins!=12:
            #     targets = targets[:, min_pitch:(min_pitch+num_output_bins)]
            # Change to use the teacher annotation as the targets
            targets = np.load(os.path.join(path_annot, fn)) > eval_thresh
            inputs_len = targets.shape[0]
            
            # Get agreed index
            agreed_index = np.load(os.path.join(path_agreed_index, fn))
            # for fn_agreed_index in os.listdir(path_agreed_index):
            #     if fn_agreed_index.startswith(fn):
            #         # Check if the alignment used for the agreement is valid (only using the train_versions)
            #         valid_alignment = True
            #         for version in val_versions + test_versions:
            #             if version in fn_agreed_index:
            #                 valid_alignment = False
            #         if valid_alignment:
            #             agreed_index = np.load(os.path.join(path_agreed_index, fn_agreed_index))

            fn_buffer_X = os.path.join(path_dataset_buffer_X, fn)
            fn_buffer_y = os.path.join(path_dataset_buffer_y, fn)
            fn_buffer_other = os.path.join(path_dataset_buffer_other, fn)

            if save_dataset_buffer:
                # Buffer segmentwise (segment length: 5000 frames)
                print(inputs.shape, targets.shape)
                seg_idx_list = range(np.ceil(inputs.shape[1] / 5000).astype(int))

                def save_segment(seg_idx):
                # for seg_idx in seg_idx_list:
                    print('Saving segment %d of %d' % (seg_idx+1, np.ceil(inputs.shape[1] / 5000).astype(int)))
                    if not os.path.exists(fn_buffer_X+'_inputs_seg_{}.npy'.format(seg_idx)):
                        np.save(fn_buffer_X+'_inputs_seg_{}.npy'.format(seg_idx), inputs[:, seg_idx*5000:(seg_idx+1)*5000, :])
                    if not os.path.exists(fn_buffer_y+'_targets_seg_{}.npy'.format(seg_idx)):
                        np.save(fn_buffer_y+'_targets_seg_{}.npy'.format(seg_idx), targets[seg_idx*5000:(seg_idx+1)*5000, :])
                
                with Pool(16) as p:
                    p.map(save_segment, seg_idx_list)

            # Add to dataset
            curr_dataset = dataset_context_fix_length_and_buffer_segmentwise(inputs=None, targets=None, params=train_dataset_params, fn=[fn_buffer_X, fn_buffer_y, fn_buffer_other], agreed_index=agreed_index, inputs_len=inputs_len)
            all_train_sets.append(curr_dataset)
            logging.info(' - file ' + str(fn) + ' added to training set.')

        if do_val:
            if any(val_version in fn for val_version in val_versions) and \
                any(val_song in fn for val_song in val_songs):
                all_val_fn.append(fn)
                if save_dataset_buffer:
                    inputs = np.transpose(np.load(os.path.join(path_data, fn)), (2, 1, 0))
                # targets = torch.from_numpy(np.load(os.path.join(path_annot, fn)).T)
                # if num_output_bins!=12:
                #     targets = targets[:, min_pitch:(min_pitch+num_output_bins)]
                # Change to use the teacher annotation as the targets
                targets = np.load(os.path.join(path_annot, fn)) > eval_thresh
                inputs_len = targets.shape[0]

                # Get agreed index
                agreed_index = np.load(os.path.join(path_agreed_index, fn))
                # for fn_agreed_index in os.listdir(path_agreed_index):
                #     if fn_agreed_index.startswith(fn):
                #         # Check if the alignment used for the agreement is valid (only using the val_versions)
                #         valid_alignment = True
                #         for version in train_versions + test_versions:
                #             if version in fn_agreed_index:
                #                 valid_alignment = False
                #         if valid_alignment:
                #             agreed_index = np.load(os.path.join(path_agreed_index, fn_agreed_index))

                fn_buffer_X = os.path.join(path_dataset_buffer_X, fn)
                fn_buffer_y = os.path.join(path_dataset_buffer_y, fn)
                fn_buffer_other = os.path.join(path_dataset_buffer_other, fn)

                if save_dataset_buffer:
                    # Save dataset buffer# Buffer segmentwise (segment length: 5000 frames)
                    print(inputs.shape, targets.shape)
                    seg_idx_list = range(np.ceil(inputs.shape[1] / 5000).astype(int))

                    def save_segment(seg_idx):
                        print('Saving segment %d of %d' % (seg_idx+1, np.ceil(inputs.shape[1] / 5000).astype(int)))
                        if not os.path.exists(fn_buffer_X+'_inputs_seg_{}.npy'.format(seg_idx)):
                            np.save(fn_buffer_X+'_inputs_seg_{}.npy'.format(seg_idx), inputs[:, seg_idx*5000:(seg_idx+1)*5000, :])
                        if not os.path.exists(fn_buffer_y+'_targets_seg_{}.npy'.format(seg_idx)):
                            np.save(fn_buffer_y+'_targets_seg_{}.npy'.format(seg_idx), targets[seg_idx*5000:(seg_idx+1)*5000, :])
                    
                    with Pool(16) as p:
                        p.map(save_segment, seg_idx_list)

                # Add to dataset
                curr_dataset = dataset_context_fix_length_and_buffer_segmentwise(inputs=None, targets=None, params=val_dataset_params, fn=[fn_buffer_X, fn_buffer_y, fn_buffer_other], agreed_index=agreed_index, eval=True, inputs_len=inputs_len)
                all_val_sets.append(curr_dataset)
                logging.info(' - file ' + str(fn) + ' added to validation set.')


    train_set = torch.utils.data.ConcatDataset(all_train_sets)
    train_loader = torch.utils.data.DataLoader(train_set, **train_params)
    logging.info('Training set & loader generated, length ' + str(len(train_set)))

    if do_val:
        val_set = torch.utils.data.ConcatDataset(all_val_sets)
        val_loader = torch.utils.data.DataLoader(val_set, **val_params)
        logging.info('Validation set & loader generated, length ' + str(len(val_set)))


# Set training configuration ###################################################

if do_train:

    criterion.to(device)

    op = optimizer_params
    if op['name']=='SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=op['initial_lr'], momentum=op['momentum'])
    elif op['name']=='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=op['initial_lr'], betas=op['betas'])
    elif op['name']=='AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=op['initial_lr'], betas=op['betas'], eps=op['eps'], weight_decay=op['weight_decay'], amsgrad=op['amsgrad'])

    sp = scheduler_params
    if sp['use_scheduler'] and sp['name']=='LambdaLR':
        start_lr, end_lr, n_decay, exp_decay = sp['start_lr'], sp['end_lr'], sp['n_decay'], sp['exp_decay']
        polynomial_decay = lambda epoch: ((start_lr - end_lr) * (1 - min(epoch, n_decay)/n_decay) ** exp_decay ) + end_lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=polynomial_decay)
    elif sp['use_scheduler'] and sp['name']=='ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=sp['mode'], \
        factor=sp['factor'], patience=sp['patience'], threshold=sp['threshold'], threshold_mode=sp['threshold_mode'], \
        cooldown=sp['cooldown'], eps=sp['eps'], min_lr=sp['min_lr'], verbose=sp['verbose'])

    ep = early_stopping_params
    if ep['use_early_stopping']:
        es = early_stopping(mode=ep['mode'], min_delta=ep['min_delta'], patience=ep['patience'], percentage=ep['percentage'])

#### START TRAINING ############################################################

    print('Starting training ...')

    logging.info('\n \n ###################### START TRAINING ###################### \n')

    # Loop over epochs
    for epoch in range(max_epochs):
        print('===============================\nEpoch {}/{}'.format(epoch+1, max_epochs))

        accum_loss, n_batches = 0, 0
        for i, (local_batch, local_labels) in enumerate(train_loader):

            print('Training step: {}/{}'.format(i+1, len(train_loader)), end='\r')
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations
            y_pred = model(local_batch)
            loss = criterion(y_pred, local_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accum_loss += loss.item()
            n_batches += 1

        print()

        train_loss = accum_loss/n_batches

        if do_val:
            accum_val_loss, n_val = 0, 0
            for j, (local_batch, local_labels) in enumerate(val_loader):

                print('Validation step: {}/{}'.format(j+1, len(val_loader)), end='\r')
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                # Model computations
                y_pred = model(local_batch)
                loss = criterion(y_pred, local_labels)

                accum_val_loss += loss.item()
                n_val += 1

            print()

            val_loss = accum_val_loss/n_val

        # Log epoch results
        if sp['use_scheduler'] and sp['name']=='LambdaLR' and do_val:
            logging.info('Epoch #' + str(epoch) + ' finished. Train Loss: ' + "{:.4f}".format(train_loss) + \
            ', Val Loss: ' + "{:.4f}".format(val_loss) + ' with lr: ' + "{:.5f}".format(scheduler.get_last_lr()[0]))
            scheduler.step()
        elif sp['use_scheduler'] and sp['name']=='ReduceLROnPlateau' and do_val:
            logging.info('Epoch #' + str(epoch) + ' finished. Train Loss: ' + "{:.4f}".format(train_loss) + \
            ', Val Loss: ' + "{:.4f}".format(val_loss) + ' with lr: ' + "{:.5f}".format(optimizer.param_groups[0]['lr']))
            scheduler.step(val_loss)
        elif sp['use_scheduler'] and sp['name']=='LambdaLR':
            logging.info('Epoch #' + str(epoch) + ' finished. Train Loss: ' + "{:.4f}".format(train_loss) + ', with lr: ' + "{:.5f}".format(scheduler.get_last_lr()[0]))
            scheduler.step()
        elif sp['use_scheduler'] and sp['name']=='ReduceLROnPlateau':
            assert False, 'Scheduler ' + sp['name'] + ' requires validation set!'
        else:
            logging.info('Epoch #' + str(epoch) + ' finished. Train Loss: ' + "{:.4f}".format(train_loss) + ', with lr: ' + "{:.5f}".format(optimizer_params['initial_lr']))

        # Perform early stopping
        if ep['use_early_stopping'] and epoch==0:
            torch.save(model.state_dict(), path_trained_model)
            logging.info('  .... model of epoch 0 saved.')
        elif ep['use_early_stopping'] and epoch>0:
            if es.curr_is_better(val_loss):
                torch.save(model.state_dict(), path_trained_model)
                logging.info('  .... model of epoch #' + str(epoch) + ' saved.')
        if ep['use_early_stopping'] and es.step(val_loss):
            break

    if not ep['use_early_stopping']:
        torch.save(model.state_dict(), path_trained_model)

    logging.info(' ### trained model saved in ' + path_trained_model + ' \n')


#### START TESTING #############################################################

if do_test:
    logging.info('\n \n ###################### START TESTING ###################### \n')
    print('\nStarting testing ...')

    # Load pretrained model
    if (not do_train) or (do_train and ep['use_early_stopping']):
        model.load_state_dict(torch.load(path_trained_model))
    if not do_train:
        logging.info(' ### trained model loaded from ' + path_trained_model + ' \n')
    model.eval()

    # Set test parameters
    half_context = test_dataset_params['context']//2

    # for test_subset in range(3):
    for test_subset in range(1):
        print('===============================\nTesting subset {}/3'.format(test_subset+1))
        if test_subset!=0:
            test_versions = test_versions_small

        n_files = 0
        total_measures = np.zeros(len(eval_measures))
        total_measures_mireval = np.zeros((14))
        n_kframes = 0 # number of frames / 10^3
        framewise_measures = np.zeros(len(eval_measures))
        framewise_measures_mireval = np.zeros((14))


        df = pd.DataFrame([])

        for test_id, fn in enumerate(os.listdir(path_data)):
            print('Checking candidate testing file {}/{}'.format(test_id+1, len(os.listdir(path_data))), end='\r')
            if any(test_version in fn for test_version in test_versions) and \
                any(test_song in fn for test_song in test_songs):

                inputs = np.transpose(np.load(os.path.join(path_data, fn)), (2, 1, 0))
                targets = np.load(os.path.join(path_annot_test, fn)).T  # use the ground truth annotation as the targets
                if num_output_bins!=12:
                    targets = targets[:, min_pitch:(min_pitch+num_output_bins)]
                if test_subset==1:
                    inputs = inputs[:, :3920, :]
                    targets = targets[:3920, :]
                inputs_context = torch.from_numpy(np.pad(inputs, ((0, 0), (half_context, half_context+1), (0, 0))))
                targets_context = torch.from_numpy(np.pad(targets, ((half_context, half_context+1), (0, 0))))

                test_set = test_set = dataset_context(inputs_context, targets_context, test_dataset_params)
                test_generator = torch.utils.data.DataLoader(test_set, **test_params)

                pred_tot = np.zeros((0, num_output_bins))
                for test_batch, test_labels in test_generator:
                    # Transfer to GPU
                    test_batch = test_batch.to(device)

                    # Model computations
                    y_pred = model(test_batch)
                    y_pred = y_pred.to('cpu')
                    # pred = torch.squeeze(y_pred).detach().numpy()
                    pred = torch.squeeze(torch.squeeze(y_pred,2),1).detach().numpy()
                    pred_tot = np.append(pred_tot, pred, axis=0)

                pred = pred_tot
                targ = targets[:targets.shape[0] - half_context*2:test_dataset_params['stride'],:]

                assert pred.shape==targ.shape, 'Shape mismatch! Target shape: '+str(targ.shape)+', Pred. shape: '+str(pred.shape)

                if not os.path.exists(os.path.join(dir_predictions)):
                    os.makedirs(os.path.join(dir_predictions))
                np.save(os.path.join(dir_predictions, fn[:-4]+'.npy'), pred)

                eval_dict = calculate_eval_measures(targ, pred, measures=eval_measures, threshold=eval_thresh, save_roc_plot=False)
                eval_numbers = np.fromiter(eval_dict.values(), dtype=float)

                metrics_mpe = calculate_mpe_measures_mireval(targ, pred, threshold=eval_thresh, min_pitch=min_pitch)
                mireval_measures = [key for key in metrics_mpe.keys()]
                mireval_numbers = np.fromiter(metrics_mpe.values(), dtype=float)

                n_files += 1
                total_measures += eval_numbers
                total_measures_mireval += mireval_numbers

                kframes = targ.shape[0]/5000
                n_kframes += kframes
                framewise_measures += kframes*eval_numbers
                framewise_measures_mireval += kframes*mireval_numbers

                res_dict = dict(zip(['Filename'] + eval_measures + mireval_measures, [fn] + eval_numbers.tolist() + mireval_numbers.tolist()))
                # df = df.append(res_dict, ignore_index=True)
                df = pd.concat([df, pd.DataFrame([res_dict])], ignore_index=True)  # after v2, pandas does not have append anymore. use concat instead

                logging.info('file ' + str(fn) + ' tested. Cosine sim: ' + str(eval_dict['cosine_sim']))

        logging.info('\n### Testing done. ################################################ \n')
        if test_subset==0:
            logging.info('#   Results for large test set (10 files) ######################### \n')
        elif test_subset==1:
            logging.info('#   Results for small test set (3 files), first 90s ############## \n')
        elif test_subset==2:
            logging.info('#   Results for small test set (3 files), full ################### \n')

        mean_measures = total_measures/n_files
        mean_measures_mireval = total_measures_mireval/n_files
        k_meas = 0
        for meas_name in eval_measures:
            logging.info('Mean ' + meas_name + ':   ' + str(mean_measures[k_meas]))
            k_meas+=1
        k_meas = 0
        for meas_name in mireval_measures:
            logging.info('Mean ' + meas_name + ':   ' + str(mean_measures_mireval[k_meas]))
            k_meas+=1

        res_dict = dict(zip(['Filename'] + eval_measures + mireval_measures, ['FILEWISE MEAN'] + mean_measures.tolist() + mean_measures_mireval.tolist()))
        # df = df.append(res_dict, ignore_index=True)
        df = pd.concat([df, pd.DataFrame([res_dict])], ignore_index=True)  # after v2, pandas does not have append anymore. use concat instead


        logging.info('\n')

        framewise_means = framewise_measures/n_kframes
        framewise_means_mireval = framewise_measures_mireval/n_kframes
        k_meas = 0
        for meas_name in eval_measures:
            logging.info('Framewise ' + meas_name + ':   ' + str(framewise_means[k_meas]))
            k_meas+=1
        k_meas = 0
        for meas_name in mireval_measures:
            logging.info('Framewise ' + meas_name + ':   ' + str(framewise_means_mireval[k_meas]))
            k_meas+=1

        res_dict = dict(zip(['Filename'] + eval_measures + mireval_measures, ['FRAMEWISE MEAN'] + framewise_means.tolist() + framewise_means_mireval.tolist()))
        # df = df.append(res_dict, ignore_index=True)
        df = pd.concat([df, pd.DataFrame([res_dict])], ignore_index=True)  # after v2, pandas does not have append anymore. use concat instead


    if test_subset==0:
        df.to_csv(path_output)
