import warnings
warnings.filterwarnings("ignore")

import argparse
import os, sys
sys.path.insert(0, os.path.join(sys.path[0], '../..'))
import numpy as np
import json
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool

from experiments.feature_preparation.utils import (
    segment_feature,
    get_feature_segment_fn,
    get_feature_segment_metainfo_fn,
)


def main(args):

    # Maestro dataset
    print('Segmenting Maestro dataset...')
    maestro_split = json.load(open('dataset_splits/maestro_split.json', 'r'))
    segment_features(
        precomputed_features_path = args.precomputed_features_path_maestro,
        segmented_features_path = args.segmented_features_path_maestro,
        pitch_folder = args.pitch_folder,
        hcqt_folder = args.hcqt_folder,
        segment_length = args.segment_length,
        fns=maestro_split['train'] + maestro_split['val'] + maestro_split['test'],
    )

    # Schubert dataset
    print('Segmenting Schubert dataset...')
    schubert_split = json.load(open('dataset_splits/schubert_split.json', 'r'))
    segment_features(
        precomputed_features_path = args.precomputed_features_path_schubert,
        segmented_features_path = args.segmented_features_path_schubert,
        pitch_folder = args.pitch_folder,
        hcqt_folder = args.hcqt_folder,
        segment_length = args.segment_length,
        fns=schubert_split['train'] + schubert_split['val'] + schubert_split['test'],
    )

    # Wagner dataset
    print('Segmenting Wagner dataset (training and validation set)...')
    wagner_split = json.load(open('dataset_splits/wagner_split.json', 'r'))
    n_versions_split = json.load(open('dataset_splits/n_versions_split.json', 'r'))
    segment_features(
        precomputed_features_path = args.precomputed_features_path_wagner,
        segmented_features_path = args.segmented_features_path_wagner,
        pitch_folder = args.pitch_folder,
        hcqt_folder = args.hcqt_folder,
        segment_length = args.segment_length,
        fns=list(set(wagner_split['train'] + wagner_split['val'] + n_versions_split['train'] + n_versions_split['val'])),
    )
    print('Segmenting Wagner dataset (testing set)...')
    segment_features(
        precomputed_features_path = args.precomputed_features_path_wagner,
        segmented_features_path = args.segmented_features_path_wagner,
        pitch_folder = args.pitch_folder_wagner_test,
        hcqt_folder = args.hcqt_folder_wagner_test,
        segment_length = args.segment_length,
        fns=wagner_split['test'],
    )

def segment_features(
        precomputed_features_path,
        segmented_features_path,
        pitch_folder,
        hcqt_folder,
        segment_length,
        fns,
    ):
    
    # Define paths
    precomputed_pitch_path = os.path.join(precomputed_features_path, pitch_folder)
    precomputed_hcqt_path = os.path.join(precomputed_features_path, hcqt_folder)
    segmented_pitch_path = os.path.join(segmented_features_path, pitch_folder)
    if not os.path.exists(segmented_pitch_path):
        os.makedirs(segmented_pitch_path)
    segmented_hcqt_path = os.path.join(segmented_features_path, hcqt_folder)
    if not os.path.exists(segmented_hcqt_path):
        os.makedirs(segmented_hcqt_path)

    # Segment pitch  TODO: udpate to process only the data in our splits.
    for fn_idx, fn in enumerate(fns):
        print('Segmenting file {}/{}'.format(fn_idx+1, len(fns)))

        # Skip if already computed
        if os.path.exists(os.path.join(segmented_hcqt_path, get_feature_segment_metainfo_fn(fn))):
            print('File {} already segmented. Skipping...'.format(fn))
            continue

        # Get precomputed features
        pitch = np.load(os.path.join(precomputed_pitch_path, fn))
        hcqt = np.load(os.path.join(precomputed_hcqt_path, fn))

        # Segment features
        all_segments_pitch, metainfo = segment_feature(pitch, segment_length, axis=1)
        all_segments_hcqt, _ = segment_feature(hcqt, segment_length, axis=1)

        # Save segments using multiprocessing
        def save_one_segment(i):
            np.save(os.path.join(segmented_pitch_path, get_feature_segment_fn(fn, i)), all_segments_pitch[i])
            np.save(os.path.join(segmented_hcqt_path, get_feature_segment_fn(fn, i)), all_segments_hcqt[i])

        pool = Pool(mp.cpu_count() // 4)
        pool.map(save_one_segment, range(len(all_segments_pitch)))

        # save metainfo in the hcqt folder
        json.dump(metainfo, open(os.path.join(segmented_hcqt_path, get_feature_segment_metainfo_fn(fn)), 'w'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Segment precomputed features')
    
    parser.add_argument('--precomputed_features_path_maestro', type=str)
    parser.add_argument('--precomputed_features_path_schubert', type=str)
    parser.add_argument('--precomputed_features_path_wagner', type=str)

    parser.add_argument('--segmented_features_path_maestro', type=str)
    parser.add_argument('--segmented_features_path_schubert', type=str)
    parser.add_argument('--segmented_features_path_wagner', type=str)

    parser.add_argument('--pitch_folder', type=str)
    parser.add_argument('--hcqt_folder', type=str)
    parser.add_argument('--pitch_folder_wagner_test', type=str)
    parser.add_argument('--hcqt_folder_wagner_test', type=str)

    parser.add_argument('--segment_length', type=int, help='Length of the segments in seconds.')
    
    args = parser.parse_args()

    main(args)