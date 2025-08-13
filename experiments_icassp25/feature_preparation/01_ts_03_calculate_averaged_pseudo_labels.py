import warnings
warnings.filterwarnings('ignore')

import argparse
import os, sys
sys.path.insert(0, os.path.join(sys.path[0], '../..'))
import numpy as np
import pandas as pd
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool

from experiments_icassp25.feature_preparation.utils import (
    get_version_pairs_fn,
    get_warping_path_folder,
    load_pseudo_label,
    extend_fn,
    segment_feature,
)


def main(args):

    # Create folder for the averaged pseudo labels & valid frames
    os.makedirs(args.path_pseudo_labels_averaged, exist_ok=True)
    os.makedirs(args.path_valid_frames, exist_ok=True)

    # Get version pairs
    version_pairs = pd.read_csv(get_version_pairs_fn(args.workspace, args.dataset))

    def process_one_pair(ipair):
        i, pair = ipair
        print('Processing version pair {}/{}'.format(i+1, len(version_pairs)))

        # Load alignment warping path
        wp = np.load(os.path.join(get_warping_path_folder(args.workspace, args.dataset), pair['fn_aligned'])).astype(int)   # (2, n_align_frames)

        # Load pseudo labels
        pseudo_labels_1 = load_pseudo_label(pair['fn1'], args, binary=True)   # (n_bins, n_frames)
        pseudo_labels_2 = load_pseudo_label(pair['fn2'], args, binary=True)   # (n_bins, n_frames)

        # Calculate average pseudo labels
        pseudo_labels_1_averaged = np.zeros(pseudo_labels_1.shape, dtype=float)
        pseudo_labels_2_averaged = np.zeros(pseudo_labels_2.shape, dtype=float)
        for idx1, idx2 in zip(wp[0], wp[1]):
            ave_label = (pseudo_labels_1[:,idx1] + pseudo_labels_2[:,idx2]) / 2
            pseudo_labels_1_averaged[:,idx1] = ave_label
            pseudo_labels_2_averaged[:,idx2] = ave_label

        # Get valid frames (non-silent frames)
        valid_frames_1 = np.where((pseudo_labels_1_averaged==1).sum(axis=0) >= 1)[0]
        valid_frames_2 = np.where((pseudo_labels_2_averaged==1).sum(axis=0) >= 1)[0]

        # Save the averaged pseudo labels
        all_segments1, _ = segment_feature(pseudo_labels_1_averaged, args.segment_length, axis=1)
        all_segments2, _ = segment_feature(pseudo_labels_2_averaged, args.segment_length, axis=1)
        for seg_idx in range(len(all_segments1)):
            np.save(os.path.join(args.path_pseudo_labels_averaged, extend_fn(pair['fn1'], seg_idx, pair['fn_aligned'])), all_segments1[seg_idx])
        for seg_idx in range(len(all_segments2)):
            np.save(os.path.join(args.path_pseudo_labels_averaged, extend_fn(pair['fn2'], seg_idx, pair['fn_aligned'])), all_segments2[seg_idx])

        # Save the valid frames
        np.save(os.path.join(args.path_valid_frames, extend_fn(pair['fn1'], fn_aligned=pair['fn_aligned'])), valid_frames_1)
        np.save(os.path.join(args.path_valid_frames, extend_fn(pair['fn2'], fn_aligned=pair['fn_aligned'])), valid_frames_2)
        
    # Process all version pairs
    ipair_list = list(version_pairs.iterrows())
    pool = Pool(mp.cpu_count() // 4)
    pool.map(process_one_pair, ipair_list)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate average pseudo labels')

    parser.add_argument('--dataset', type=str, help='Name of the dataset, schubert or wagner')
    parser.add_argument('--workspace', type=str, help='Path to the workspace')
    parser.add_argument('--path_x', type=str, help='Path to the input feature')
    parser.add_argument('--path_pseudo_labels', type=str, help='Path to the pseudo labels')
    parser.add_argument('--path_pseudo_labels_averaged', type=str, help='Path to the average pseudo labels')
    parser.add_argument('--path_valid_frames', type=str, help='Path to the valid (non-silent) frames')

    parser.add_argument('--threshold', type=float, help='Threshold for binarise the pseudo labels', default=0.5)
    parser.add_argument('--segment_length', type=int, help='Length of the segments in seconds.')

    args = parser.parse_args()

    main(args)





