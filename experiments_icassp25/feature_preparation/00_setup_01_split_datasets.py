import warnings
warnings.filterwarnings("ignore")

import argparse
import os, sys
sys.path.insert(0, os.path.join(sys.path[0], '../..'))
import json
import pandas as pd

from dataset_splits import (
    train_songs_schubert, val_songs_schubert, test_songs_schubert,
    train_versions_schubert, val_versions_schubert, test_versions_schubert,
    train_songs_wagner, val_songs_wagner, test_songs_wagner,
    train_versions_wagner, val_versions_wagner, test_versions_wagner,
    train_n_versions, val_n_versions,
)


def main(args):

    if not os.path.exists('dataset_splits'):
        os.makedirs('dataset_splits')

    maestro_split(args)
    schubert_neither_split(args)
    wagner_neither_split(args)
    n_versions_split(args)


def maestro_split(args):

    # Get train/val/test split from the original metadata
    metadata = pd.read_csv(os.path.join(args.original_dataset_path_maestro, 'maestro-v3.0.0.csv'))
    # Get all filenames in the pitch folder as a set (for faster lookup)
    all_fn = set(os.listdir(os.path.join(args.precomputed_features_path_maestro, args.pitch_folder)))

    # Initialise splits
    maestro_split = {'train': [], 'val': [], 'test': []}

    for i, row in metadata.iterrows():
        if row['audio_filename'][5:-4]+'.npy' in all_fn:
            if row['split'] == 'train':
                maestro_split['train'].append(row['audio_filename'][5:-4]+'.npy')
            elif row['split'] == 'validation':
                maestro_split['val'].append(row['audio_filename'][5:-4]+'.npy')
            elif row['split'] == 'test':
                maestro_split['test'].append(row['audio_filename'][5:-4]+'.npy')
                
    # Save splits into json files
    json.dump(maestro_split, open('dataset_splits/maestro_split.json', 'w'))


def schubert_neither_split(args):

    # Get data path
    data_path = os.path.join(args.precomputed_features_path_schubert, args.pitch_folder)

    # Initialise splits
    schubert_split = {'train': [], 'val': [], 'test': []}

    # Populate splits
    for fn in os.listdir(data_path):
        if any(train_version in fn for train_version in train_versions_schubert) and any(train_song in fn for train_song in train_songs_schubert):
            schubert_split['train'].append(fn)
        elif any(val_version in fn for val_version in val_versions_schubert) and any(val_song in fn for val_song in val_songs_schubert):
            schubert_split['val'].append(fn)
        elif any(test_version in fn for test_version in test_versions_schubert) and any(test_song in fn for test_song in test_songs_schubert):
            schubert_split['test'].append(fn)

    # Save splits into json files
    json.dump(schubert_split, open('dataset_splits/schubert_split.json', 'w'))


def wagner_neither_split(args):

    # Get data path
    data_path = os.path.join(args.precomputed_features_path_wagner, args.pitch_folder)
    data_path_test = os.path.join(args.precomputed_features_path_wagner, args.pitch_folder_wagner_test)

    # Initialise splits
    wagner_split = {'train': [], 'val': [], 'test': []}

    # Populate splits
    for fn in os.listdir(data_path) + os.listdir(data_path_test):
        if any(train_version in fn for train_version in train_versions_wagner) and any(train_song in fn for train_song in train_songs_wagner):
            wagner_split['train'].append(fn)
        elif any(val_version in fn for val_version in val_versions_wagner) and any(val_song in fn for val_song in val_songs_wagner):
            wagner_split['val'].append(fn)
        elif any(test_version in fn for test_version in test_versions_wagner) and any(test_song in fn for test_song in test_songs_wagner):
            wagner_split['test'].append(fn)

    # Save splits into json files
    json.dump(wagner_split, open('dataset_splits/wagner_split.json', 'w'))


def n_versions_split(args):

    # Get data path
    data_path = os.path.join(args.precomputed_features_path_wagner, args.pitch_folder)
    data_path_test = os.path.join(args.precomputed_features_path_wagner, args.pitch_folder_wagner_test)

    # Initialise splits
    n_versions_split = {'train': [], 'val': [], 'test': []}

    # Populate splits
    for fn in os.listdir(data_path) + os.listdir(data_path_test):
        if any(train_version in fn for train_version in train_n_versions) and any(train_song in fn for train_song in train_songs_wagner):
            n_versions_split['train'].append(fn)
        elif any(val_version in fn for val_version in val_n_versions) and any(val_song in fn for val_song in val_songs_wagner):
            n_versions_split['val'].append(fn)
        elif any(test_version in fn for test_version in test_versions_wagner) and any(test_song in fn for test_song in test_songs_wagner):
            n_versions_split['test'].append(fn)

    # Save splits into json files
    json.dump(n_versions_split, open('dataset_splits/n_versions_split.json', 'w'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Split datasets and save the splits in csv files')

    parser.add_argument('--original_dataset_path_maestro', type=str, required=True)
    parser.add_argument('--precomputed_features_path_maestro', type=str, required=True)
    parser.add_argument('--precomputed_features_path_schubert', type=str, required=True)
    parser.add_argument('--precomputed_features_path_wagner', type=str, required=True)
    parser.add_argument('--pitch_folder', type=str, required=True)
    parser.add_argument('--pitch_folder_wagner_test', type=str, required=True)

    args = parser.parse_args()

    main(args)

