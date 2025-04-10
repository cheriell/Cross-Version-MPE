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
from libdl.data_loaders.hcqt_datasets_onthefly import dataset_context
from libdl.nn_models import deep_cnn_segm_sigmoid
from libdl.metrics import early_stopping, calculate_eval_measures, calculate_mpe_measures_mireval
import logging


################################################################################
#### Set experimental configuration ############################################
################################################################################

dataset = 'Schubert_Winterreise'

# Get experiment name from script name
curr_filepath = sys.argv[0]
expname = curr_filename = os.path.splitext(os.path.basename(curr_filepath))[0]
print(' ... running experiment ' + expname)

# Which steps to perform
do_train = False
do_val = False
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
                      'stride': 50,
                      'compression': 10
                      }
test_dataset_params = {'context': 75,
                       'stride': 1,
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
test_params = {'batch_size': 1, #50,
              'shuffle': False,
              'num_workers': 0, #8
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

    max_epochs = 100

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
path_annot = os.path.join('/mnt/d/workspace/cross-version-mpe/feature_preparation', dataset, 'teacher_annotations')

# Where to save models
# dir_models = os.path.join(os.sep, 'tsi', 'clusterhome', 'cweiss', 'models')
dir_models = os.path.join('/mnt/d/workspace/cross-version-mpe/experiments', dataset, 'models')
if not os.path.exists(dir_models):
    os.makedirs(dir_models)
fn_model = expname + '.pt'
# path_trained_model = os.path.join(dir_models, fn_model)
path_trained_model = os.path.join(dir_models, 'teacher_maestro.pt')  # use the teacher model trained on maestro

# Where to save results
# dir_output = os.path.join(os.sep, 'tsi', 'clusterhome', 'cweiss', 'code', 'deep_pitch_estimation', 'experiments', 'results_filewise')
dir_output = os.path.join('/mnt/d/workspace/cross-version-mpe/experiments', dataset, 'results_filewise')
if not os.path.exists(dir_output):
    os.makedirs(dir_output)
fn_output = expname + '.csv'
path_output = os.path.join(dir_output, fn_output)

# Where to save predictions
# dir_predictions = os.path.join(os.sep, 'tsi', 'clusterhome', 'cweiss', 'predictions', expname)
dir_predictions = os.path.join('/mnt/d/workspace/cross-version-mpe/experiments', dataset, 'predictions', expname)
if not os.path.exists(dir_predictions):
    os.makedirs(dir_predictions)

# Where to save logs
fn_log = expname + '.txt'
# path_log = os.path.join(os.sep, 'tsi', 'clusterhome', 'cweiss', 'code', 'deep_pitch_estimation', 'experiments', 'logs', fn_log)
path_log = os.path.join('/mnt/d/workspace/cross-version-mpe/experiments', dataset, 'logs', fn_log)
if not os.path.exists(os.path.dirname(path_log)):
    os.makedirs(os.path.dirname(path_log))

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

logging.info(' --- Model config: -------------------------------------------- ')
logging.info('Model: ' + model.__class__.__name__)
logging.info("Model parameters: {0}".format(model_params))
logging.info('\n' + str(summary(model, input_size=(1, 6, 174, 216))))

# Generate training dataset ####################################################
if do_val:
    assert do_train, 'Validation without training not possible!'
train_songs = ['D911-01', 'D911-02', 'D911-03', 'D911-04', 'D911-05', 'D911-06', 'D911-07', 'D911-08', 'D911-09', 'D911-10', 'D911-11', 'D911-12', 'D911-13', ]
val_songs = ['D911-14', 'D911-15', 'D911-16', ]
test_songs = ['D911-17', 'D911-18', 'D911-19', 'D911-20', 'D911-21', 'D911-22', 'D911-23', 'D911-24']
train_versions = ['AL98', 'FI55', 'FI80', 'OL06', 'QU98']
val_versions = ['FI66', 'TR99']
test_versions = ['HU33', 'SC06']
test_versions_small = test_versions   # Set it to be the same as test_versions

all_train_fn = []
all_train_sets = []
all_val_fn = []
all_val_sets = []

path_dataset_buffer = os.path.join('/mnt/d/workspace/cross-version-mpe/experiments', dataset, 'dataset_buffer', expname)
if not os.path.exists(path_dataset_buffer):
    os.makedirs(path_dataset_buffer)

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

    for test_subset in range(3):
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

                test_set = dataset_context(inputs_context, targets_context, test_dataset_params, fn=os.path.join(path_dataset_buffer, fn))
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
                targ = targets

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

                kframes = targ.shape[0]/1000
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