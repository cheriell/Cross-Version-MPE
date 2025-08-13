import warnings
warnings.filterwarnings("ignore")

import argparse
import os, sys
sys.path.insert(0, os.path.join(sys.path[0], '../..'))
import pandas as pd
import numpy as np
from functools import reduce
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool

from experiments_icassp25.feature_preparation.utils import get_warping_path_folder, get_version_pairs_fn


def main(args):

    calculate_warping_paths(args, 'schubert')
    calculate_warping_paths(args, 'wagner')
    calculate_warping_paths(args, 'n_versions')
    
    
def calculate_warping_paths(args, dataset):

    # Arguments
    workspace = args.workspace
    precomputed_features_path_schubert = args.precomputed_features_path_schubert
    precomputed_features_path_wagner = args.precomputed_features_path_wagner
    hcqt_folder = args.hcqt_folder

    # Define paths
    if dataset == 'schubert':
        feature_folder = os.path.join(precomputed_features_path_schubert, hcqt_folder)
    elif dataset == 'wagner':
        feature_folder = os.path.join(precomputed_features_path_wagner, hcqt_folder)
    elif dataset == 'n_versions':
        feature_folder = os.path.join(precomputed_features_path_wagner, hcqt_folder)

    if dataset == 'schubert':
        wp_folder = get_warping_path_folder(workspace, 'schubert')
    elif dataset == 'wagner' or dataset == 'n_versions':
        wp_folder = get_warping_path_folder(workspace, 'wagner')
    if not os.path.exists(wp_folder):
        os.makedirs(wp_folder)

    # Get version pairs
    df_pairs = pd.read_csv(get_version_pairs_fn(workspace, dataset))

    # Process one pair
    def process_one_pair(ipair):
        i, pair = ipair
        print('Processing pair %d of %d' % (i+1, len(df_pairs)))

        fn1 = pair['fn1']
        fn2 = pair['fn2']
        fn_aligned = pair['fn_aligned']

        # Skip if already processed
        if os.path.exists(os.path.join(wp_folder, fn_aligned)):
            print('Already processed, skipping')
            return

        # load hcqt features for both versions
        f_hcqt1 = np.load(os.path.join(feature_folder, fn1))
        f_hcqt2 = np.load(os.path.join(feature_folder, fn2))

        # reduce hcqt features to chroma features (36 bins per octave, hcqt shape: (n_bins, n_frames, n_harmonics))
        f_octave1 = reduce(lambda x, y: x+y, [f_hcqt1[i:i+36, :, 0] for i in range(6)])
        f_octave2 = reduce(lambda x, y: x+y, [f_hcqt2[i:i+36, :, 0] for i in range(6)])
        f_chroma1 = reduce(lambda x, y: x+y, [f_octave1[i::3, :] for i in range(3)])
        f_chroma2 = reduce(lambda x, y: x+y, [f_octave2[i::3, :] for i in range(3)])
        print('Chroma 1 shape:', f_chroma1.shape)
        print('Chroma 2 shape:', f_chroma2.shape)
        
        alignment = sync_via_mrmsdtw(f_chroma1=f_chroma1,
                        f_chroma2=f_chroma2,
                        input_feature_rate=50,
                        verbose=False)
        print(alignment.shape)
        
        np.save(os.path.join(wp_folder, fn_aligned), alignment)

    # Process all pairs
    ipair_list = list(df_pairs.iterrows())
    pool = Pool(mp.cpu_count() // 4)
    pool.map(process_one_pair, ipair_list)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate warping paths for all version pairs')

    parser.add_argument('--workspace', type=str)
    parser.add_argument('--precomputed_features_path_schubert', type=str)
    parser.add_argument('--precomputed_features_path_wagner', type=str)
    parser.add_argument('--hcqt_folder', type=str)

    args = parser.parse_args()

    main(args)