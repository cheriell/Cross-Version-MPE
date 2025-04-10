import warnings
warnings.filterwarnings("ignore")

import os, sys
sys.path.insert(0, os.path.join(sys.path[0], '../..'))
import argparse
import pandas as pd
import numpy as np
import json
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool

from experiments.feature_preparation.utils import (
    get_version_pairs_fn,
    get_warping_path_folder,
    get_consistent_frames_fn,
    get_feature_segment_fn,
    get_feature_segment_metainfo_fn,
    load_pseudo_label
)


bi_threshold = 0.5   # binary threshold for getting the pseudo ground truth

def main(args):

    # Create the directory of the consistent frames if it does not exist
    os.makedirs(args.path_consistent_frames, exist_ok=True)

    # Get version pairs
    version_pairs = pd.read_csv(get_version_pairs_fn(args.workspace, args.dataset))
    
    # Find consistent frames
    if args.consistency_type == 'hard':
        hard_consistency(version_pairs, args)
    elif args.consistency_type == 'soft':
        soft_consistency(version_pairs, args)
    else:
        raise ValueError("Invalid consistency type")


def hard_consistency(version_pairs, args):

    def process_one_pair(ipair):
        i, pair = ipair
        print('Processing version pair {}/{}'.format(i+1, len(version_pairs)))

        consistent_frames_1_fn = os.path.join(args.path_consistent_frames, get_consistent_frames_fn(pair['fn1'], pair['fn_aligned']))
        consistent_frames_2_fn = os.path.join(args.path_consistent_frames, get_consistent_frames_fn(pair['fn2'], pair['fn_aligned']))
        # Skip if the consistent frames already exist
        if os.path.exists(consistent_frames_1_fn) and os.path.exists(consistent_frames_2_fn):
            return

        # Load alignment warping path
        wp = np.load(os.path.join(get_warping_path_folder(args.workspace, args.dataset), pair['fn_aligned'])).astype(int)  # (2, n_align_frames)

        # Load pseudo labels
        pseudo_labels_1 = load_pseudo_label(pair['fn1'], args)   # (n_bins, n_frames)
        pseudo_labels_2 = load_pseudo_label(pair['fn2'], args)   # (n_bins, n_frames)
        assert wp[0, -1] == pseudo_labels_1.shape[1] - 1, 'wp[0, -1]: {}, pseudo_labels_1.shape[1]: {}, pair: \n{}'.format(wp[0, -1], pseudo_labels_1.shape[1], pair)
        assert wp[1, -1] == pseudo_labels_2.shape[1] - 1, 'wp[1, -1]: {}, pseudo_labels_2.shape[1]: {}, pair: \n{}'.format(wp[1, -1], pseudo_labels_2.shape[1], pair)

        # Get the consistent frames
        consistent_frames_1, consistent_frames_2 = set(), set()

        for idx1, idx2 in zip(wp[0,:], wp[1,:]):
            # Check version 1 consistency
            if pseudo_labels_1[:,idx1].sum() > 0 and idx1 not in consistent_frames_1:   # ignore silent frames
                consistent = False
                for j2 in range(max(0, idx2-args.align_tolerance), min(pseudo_labels_2.shape[1], idx2+args.align_tolerance+1)):
                    if np.sum((pseudo_labels_1[:,idx1] == pseudo_labels_2[:,j2]) == False) <= args.allow_diff:
                        consistent = True
                        break
                if consistent:
                    consistent_frames_1.add(idx1)
            # Check version 2 consistency
            if pseudo_labels_2[:,idx2].sum() > 0 and idx2 not in consistent_frames_2:   # ignore silent frames
                consistent = False
                for j1 in range(max(0, idx1-args.align_tolerance), min(pseudo_labels_1.shape[1], idx1+args.align_tolerance+1)):
                    if np.sum((pseudo_labels_1[:,j1] == pseudo_labels_2[:,idx2]) == False) <= args.allow_diff:
                        consistent = True
                        break
                if consistent:
                    consistent_frames_2.add(idx2)

        # Save the consistent frames for each pair
        np.save(consistent_frames_1_fn, np.array(list(consistent_frames_1)))
        np.save(consistent_frames_2_fn, np.array(list(consistent_frames_2)))

    # Process each pair using multiprocessing
    ipair_list = list(version_pairs.iterrows())
    pool = Pool(mp.cpu_count() // 4)
    pool.map(process_one_pair, ipair_list)

    # Merge the consistent frames per file
    splits = json.load(open('dataset_splits/{}_split.json'.format(args.dataset), 'r'))
    fns = splits['train'] + splits['val']
    for fn in fns:
        consistent_frames = set()
        for _, pair in version_pairs.iterrows():
            if pair['fn1'] == fn or pair['fn2'] == fn:
                consistent_frames = consistent_frames.union(set(np.load(os.path.join(args.path_consistent_frames, get_consistent_frames_fn(fn, pair['fn_aligned'])))))
        consistent_frames = np.array(list(consistent_frames))
        consistent_frames = np.sort(consistent_frames)
        np.save(os.path.join(args.path_consistent_frames, get_consistent_frames_fn(fn)), consistent_frames)

    # Print statistics
    if args.print_stats:
        _print_statistics(fns, args)


def soft_consistency(version_pairs, args):

    # Create a temporary folder to save the metrics
    temp_metrics_folder = os.path.join(args.workspace, 'temp_metrics')
    os.makedirs(temp_metrics_folder, exist_ok=True)
    get_metric_fn = lambda fn, fn_aligned: os.path.join(temp_metrics_folder, '{}_by_pair_{}'.format(fn, fn_aligned))
    
    # Calculate metric distribution over all version pairs
    def process_one_pair(ipair):
        i, pair = ipair
        print('Processing version pair {}/{}'.format(i+1, len(version_pairs)))

        # Load alignment warping path
        wp = np.load(os.path.join(get_warping_path_folder(args.workspace, args.dataset), pair['fn_aligned'])).astype(int)  # (2, n_align_frames)

        # Load pseudo labels
        pseudo_labels_1 = load_pseudo_label(pair['fn1'], args, binary=False)   # (n_bins, n_frames)
        pseudo_labels_2 = load_pseudo_label(pair['fn2'], args, binary=False)   # (n_bins, n_frames)

        # Initialize the metric array with maximum value (the smaller the better)
        metrics_1 = np.ones(pseudo_labels_1.shape[1]) * np.inf
        metrics_2 = np.ones(pseudo_labels_2.shape[1]) * np.inf

        for idx1, idx2 in zip(wp[0,:], wp[1,:]):
            # Calculate the metric for version 1
            if pseudo_labels_1[:,idx1].sum() > 0:   # ignore silent frames
                for j2 in range(max(0, idx2-args.align_tolerance), min(pseudo_labels_2.shape[1], idx2+args.align_tolerance+1)):
                    metric = _calculate_soft_consistency_metric(pseudo_labels_1[:,idx1], pseudo_labels_2[:,j2], args.metric)
                    if metric < metrics_1[idx1]:
                        metrics_1[idx1] = metric
            # Calculate the metric for version 2
            if pseudo_labels_2[:,idx2].sum() > 0:   # ignore silent frames
                for j1 in range(max(0, idx1-args.align_tolerance), min(pseudo_labels_1.shape[1], idx1+args.align_tolerance+1)):
                    metric = _calculate_soft_consistency_metric(pseudo_labels_1[:,j1], pseudo_labels_2[:,idx2], args.metric)
                    if metric < metrics_2[idx2]:
                        metrics_2[idx2] = metric

        # Save the metrics
        np.save(get_metric_fn(pair['fn1'], pair['fn_aligned']), metrics_1)
        np.save(get_metric_fn(pair['fn2'], pair['fn_aligned']), metrics_2)

    # Process each pair using multiprocessing
    ipair_list = list(version_pairs.iterrows())
    # pool = Pool(mp.cpu_count() // 4)
    pool = Pool(1)
    pool.map(process_one_pair, ipair_list[:5])

    # Calculate the metric distribution over all version pairs
    metrics_all = np.zeros(0)
    for _, pair in version_pairs.iterrows():
        metrics_1 = np.load(get_metric_fn(pair['fn1'], pair['fn_aligned']))
        metrics_2 = np.load(get_metric_fn(pair['fn2'], pair['fn_aligned']))
        metrics_all = np.concatenate([metrics_all, metrics_1, metrics_2])
    metrics_all.sort()
    cutoff_value = metrics_all[int(len(metrics_all) * args.cutoff_ratio)]

    # Print statistics
    if args.print_stats:
        print('Cutoff value: {:.4f}'.format(cutoff_value))
        print('Metrics distribution: \nmean: {:.4f}, std: {:.4f} median: {:.4f}'.format(metrics_all.mean(), metrics_all.std(), np.median(metrics_all)))

    # Save the consistent frames for each file
    splits = json.load(open('dataset_splits/{}_split.json'.format(args.dataset), 'r'))
    fns = splits['train'] + splits['val']
    for fn in fns:
        consistent_frames = set()
        for _, pair in version_pairs.iterrows():
            if pair['fn1'] == fn or pair['fn2'] == fn:
                metrics = np.load(get_metric_fn(fn, pair['fn_aligned']))
                consistent_frames = consistent_frames.union(set(np.where(metrics < cutoff_value)[0]))
        consistent_frames = np.array(list(consistent_frames))
        consistent_frames = np.sort(consistent_frames)
        np.save(os.path.join(args.path_consistent_frames, get_consistent_frames_fn(fn)), consistent_frames)

    # Print statistics
    if args.print_stats:
        _print_statistics(fns, args)



def _calculate_soft_consistency_metric(labels1, labels2, metric='cosine'):
    if metric == 'euclidean':
        metric = np.linalg.norm(labels1 - labels2)
    elif metric == 'cosine':
        metric = np.dot(labels1, labels2) / (np.linalg.norm(labels1) * np.linalg.norm(labels2))
        metric = 1 - metric  # smaller the better
    else:
        raise ValueError("Invalid metric")
    return metric


def _print_statistics(fns, args):
    length_sum, consistent_frames_sum = 0, 0
    for fn in fns:
        consistent_frames = np.load(os.path.join(args.path_consistent_frames, get_consistent_frames_fn(fn)))
        length_sum += int(json.load(open(os.path.join(args.path_x, get_feature_segment_metainfo_fn(fn)), 'r'))['length'])
        consistent_frames_sum += len(consistent_frames)
    print('Percentage of consistent frames: {:.4f}%'.format(consistent_frames_sum / length_sum * 100))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Find consistent time frames between versions in the dataset")
    
    # Add arguments
    parser.add_argument('--dataset', type=str, help="Name of the dataset to use, schubert or wagner", required=True)
    parser.add_argument('--workspace', type=str, help="Path to the workspace", required=True)
    parser.add_argument('--path_x', type=str, help="Path to the segmented input features", required=True)
    parser.add_argument('--path_pseudo_labels', type=str, help="Path to the pseudo labels", required=True)
    parser.add_argument('--path_consistent_frames', type=str, help="Path to save the consistent frames", required=True)
    parser.add_argument('--print_stats', action='store_true', help="Print statistics of the consistent frames")

    # Add subparsers for different consistency types
    subparsers = parser.add_subparsers(dest='consistency_type', help='Hard consistency options')

    # Subparser for hard consistency
    parser_hard = subparsers.add_parser('hard')
    parser_hard.add_argument('--threshold', type=float, help='Threshold for binarising the pseudo labels', default=0.5)
    parser_hard.add_argument('--allow_diff', type=int, help='Allowable difference in the number of inconsistent pitches', default=2)
    parser_hard.add_argument('--align_tolerance', type=int, help='Allow for +-N frames for soft alignment', default=2)

    # Subparser for soft consistency
    parser_soft = subparsers.add_parser('soft')
    parser_soft.add_argument('--cutoff_ratio', type=float, help='Ratio of frames that have better consistency amonge all version pairs, which we use to train the model', default=0.5)
    parser_soft.add_argument('--threshold', type=float, help='Threshold for binarising the pseudo labels', default=0.4)
    parser_soft.add_argument('--metric', type=str, help="Metric to use for calculating the consistency between versions, can be 'euclidean' or 'cosine'", default='cosine')
    parser_soft.add_argument('--align_tolerance', type=int, help='Allow for +-N frames for soft alignment, the consistency metric will then be calculated as the best between the +-N frames', default=2)

    args = parser.parse_args()

    main(args)