import numpy as np
import os
import json
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool


########################################################
# Feature segmentation
########################################################
def segment_feature(feature, segment_length, axis=1):
    """Segments a feature into segments of length segment_length.
    """
    length = feature.shape[axis]
    n_segments = np.ceil(length / (segment_length * 100)).astype(int)

    all_segments = []
    for i in range(n_segments):
        start = i * segment_length * 100
        end = (i + 1) * segment_length * 100
        segment = feature[:, start:end]
        all_segments.append(segment)

    metainfo = {
        'n_segments': str(n_segments),
        'segment_length': str(segment_length),
        'length': str(length),
    }

    return all_segments, metainfo

def get_feature_segment_fn(orig_filename, segment_id):
    """Returns the filename of a feature segment.
    """
    return orig_filename + '_segment_{}.npy'.format(segment_id)

def get_feature_segment_metainfo_fn(orig_filename):
    """Returns the filename of a feature segment.
    """
    return orig_filename + '_metainfo.json'

def get_x_context_by_segment_idx(path_x, fn, seg_idx, half_context, n_segments):
    
    x = np.load(os.path.join(path_x, get_feature_segment_fn(fn, seg_idx)))  # (n_bins, n_frames, n_chan)
    if seg_idx == 0:
        context_left = np.zeros((x.shape[0], half_context, x.shape[2]))
    else:
        context_left = np.load(os.path.join(path_x, get_feature_segment_fn(fn, seg_idx-1)))[:, -half_context:, :]
    if seg_idx == n_segments-1:
        context_right = np.zeros((x.shape[0], half_context, x.shape[2]))
    else:
        context_right = np.load(os.path.join(path_x, get_feature_segment_fn(fn, seg_idx+1)))[:, :half_context, :]
        # pad the context_right if it is shorter than half_context
        if context_right.shape[1] < half_context:
            context_right = np.pad(context_right, ((0, 0), (0, half_context - context_right.shape[1]), (0, 0)), 'constant', constant_values=0)
    x_context = np.concatenate([context_left, x, context_right], axis=1)  # (n_bins, n_frames, n_chan)

    return x_context


########################################################
# Version pairs/groups and warping paths
########################################################
def get_version_pairs_fn(workspace, dataset):
    return os.path.join(workspace, 'version_pairs', 'version_pairs_{}.csv'.format(dataset))
    
def get_fn_by_song_version(fns, song, version):
    for fn in fns:
        if song in fn and version in fn:
            return fn

def get_song_version_by_fn(fn, dataset):
    if dataset == 'wagner':
        _, song, _, version = fn[:-4].split('_')
    else:
        raise NotImplementedError
    return song, version
        
def get_warping_path_folder(workspace, dataset):
    return os.path.join(workspace, 'warping_paths', dataset)

def get_consistent_frames_fn(fn, fn_aligned=None):
    if fn_aligned is not None:
        return '{}_by_alignment_{}'.format(fn, fn_aligned)
    else:
        return fn

def get_all_matched_fns_filename(workspace, n_versions, split):
    return os.path.join(workspace, 'experiments/n_versions/all_matched_fns', 'all_matched_fns-{}_versions-{}.json'.format(n_versions, split))
    

########################################################
# Pseudo labels
########################################################
def load_pseudo_label(fn_noSegmentId, args, binary=True):
    """
    Load segmented pseudo labels as a whole.
    Parameters
    ----------
    fn_noSegmentId : str
        The filename of the pseudo label without the segment index tail.
    args: argparse.Namespace
        The arguments.
    binary : bool
        If True, the pseudo labels are binarized by a threshold.
    """
    fn = fn_noSegmentId.split('_by_alignment_')[0]
    n_segments = int(json.load(open(os.path.join(args.path_x, get_feature_segment_metainfo_fn(fn)), 'r'))['n_segments'])
    pseudo_labels_raw = np.concatenate([np.load(os.path.join(args.path_pseudo_labels, get_feature_segment_fn(fn_noSegmentId, i))) for i in range(n_segments)], axis=1)   # (n_bins, n_frames)
    if binary:
        pseudo_labels = (pseudo_labels_raw > args.threshold).astype(int)   # (n_bins, n_frames)
    else:
        pseudo_labels = pseudo_labels_raw
    return pseudo_labels


########################################################
# fn operations
########################################################
def extend_fn(fn, segment_id=None, fn_aligned=None):
    if segment_id is not None:
        if fn_aligned is not None:
            return '{}_by_alignment_{}'.format(fn, fn_aligned)  + '_segment_{}.npy'.format(segment_id)
        else:
            return fn + '_segment_{}.npy'.format(segment_id)
    else:
        if fn_aligned is not None:
            return '{}_by_alignment_{}'.format(fn, fn_aligned)
        else:
            return fn

def get_fn_aligned(song, version1, version2):
    fn_aligned = '_'.join([song, version1, version2]) + '.npy'
    return fn_aligned

def extend_fn_by_version_group(fn, group_label, segment_id=None):
    if segment_id is not None:
        return "{}_{}.npy_segment_{}.npy".format(fn, group_label, segment_id)
    else:
        return "{}_{}.npy".format(fn, group_label)


########################################################
# Version groups
########################################################
    
def filter_version_groups(n_groups):
    """Filter the version groups to keep only 11 groups per file.
    Otherwise, the pseudo labels will be too large due to many combinations of version groups.

    Parameters
    ----------
    n_groups : int
        The number of version groups.
    
    Returns 
    -------
    list
        The filtered index of version groups.
    """
    n_groups_max = 10
    if n_groups > n_groups_max:
        hop = n_groups / n_groups_max
        return [int(round(i * hop)) for i in range(n_groups_max)]
    else:
        return list(range(n_groups))
