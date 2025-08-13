import warnings
warnings.filterwarnings("ignore")

import argparse
import os, sys
sys.path.insert(0, os.path.join(sys.path[0], '../..'))
import pandas as pd
import json

from experiments_icassp25.feature_preparation.utils import get_version_pairs_fn, get_fn_by_song_version, get_fn_aligned
from dataset_splits import (
    train_songs_wagner, train_versions_wagner,
    val_songs_wagner, val_versions_wagner,
    train_n_versions, val_n_versions
)


def main(args):

    vp_folder = os.path.dirname(get_version_pairs_fn(args.workspace, 'schubert'))
    if not os.path.exists(vp_folder):
        os.makedirs(vp_folder)

    find_version_pairs_for_schubert(args)
    find_version_pairs_for_wagner(args, dataset_str='wagner')
    find_version_pairs_for_wagner(args, dataset_str='n_versions')


def find_version_pairs_for_schubert(args):
    print('Finding version pairs for Schubert dataset')

    # get global key annotations
    ann_audio_global_key = pd.read_csv(os.path.join(args.original_dataset_path_schubert, 'ann_audio_globalkey.csv'), sep=';')
    # Get all work IDs for the compositions
    all_workIDs = ann_audio_global_key['WorkID'].unique()

    # Initialise list of version pairs
    pairs = []

    # For each work ID, find all version pairs
    for workID in all_workIDs:
        df = ann_audio_global_key[ann_audio_global_key['WorkID'] == workID]
        unique_keys = df['key'].unique()

        for key in unique_keys:
            df = df[df['key'] == key]
            if len(df) == 1:
                continue

            versions = df['PerformanceID'].unique()

            # find all possible pairs of versions
            for i in range(len(versions)-1):
                for j in range(i+1, len(versions)):

                    pair = {
                        'WorkID': workID, 
                        'key': key,
                        'PerformanceID1': versions[i],
                        'PerformanceID2': versions[j],
                        'fn1': '_'.join([workID, versions[i]])+'.npy',
                        'fn2': '_'.join([workID, versions[j]])+'.npy',
                        'fn_aligned': get_fn_aligned(workID, versions[i], versions[j]),
                    }

                    pairs.append(pair)

    print('Found %d pairs of versions' % len(pairs))

    # Filter the version pairs to only include those that are in the same dataset split (train or val)
    schubert_split = json.load(open('dataset_splits/schubert_split.json', 'r'))
    fns_train = set(schubert_split['train'])
    fns_val = set(schubert_split['val'])
    
    pairs_filtered = []
    for pair in pairs:
        if pair['fn1'] in fns_train and pair['fn2'] in fns_train:
            pairs_filtered.append(pair)
        elif pair['fn1'] in fns_val and pair['fn2'] in fns_val:
            pairs_filtered.append(pair)

    # Save pairs to csv
    df_pairs = pd.DataFrame(pairs_filtered)
    df_pairs.to_csv(get_version_pairs_fn(args.workspace, 'schubert'), index=False)


def find_version_pairs_for_wagner(args, dataset_str='wagner'):
    print('Finding version pairs for Wagner dataset')

    # Initialise list of version pairs
    pairs = []
    
    # Get all fns in the training and validation set
    wagner_split = json.load(open('dataset_splits/{}_split.json'.format(dataset_str), 'r'))
    fns = wagner_split['train'] + wagner_split['val']

    def update_pairs_by_split(songs, versions):
        for song in songs:
            for i in range(len(versions)-1):
                for j in range(i+1, len(versions)):
                    pair = {
                        'WorkID': song,
                        'PerformanceID1': versions[i],
                        'PerformanceID2': versions[j],
                        'fn1': get_fn_by_song_version(fns, song, versions[i]),
                        'fn2': get_fn_by_song_version(fns, song, versions[j]),
                        'fn_aligned': get_fn_aligned(song, versions[i], versions[j]),
                    }

                    pairs.append(pair)

    if dataset_str == 'wagner':
        update_pairs_by_split(train_songs_wagner, train_versions_wagner)
        update_pairs_by_split(val_songs_wagner, val_versions_wagner)
    elif dataset_str == 'n_versions':
        update_pairs_by_split(train_songs_wagner, train_n_versions)
        update_pairs_by_split(val_songs_wagner, val_n_versions)

    print('Found %d pairs of versions' % len(pairs))

    # Save pairs to csv
    df_pairs = pd.DataFrame(pairs)
    df_pairs.to_csv(get_version_pairs_fn(args.workspace, dataset_str), index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--original_dataset_path_schubert', type=str)
    parser.add_argument('--workspace', type=str)

    args = parser.parse_args()

    main(args)

