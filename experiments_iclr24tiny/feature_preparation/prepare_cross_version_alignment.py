import os, sys
sys.path.insert(0, os.path.join(sys.path[0], '../../'))
import pandas as pd
import numpy as np
from functools import reduce
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
import matplotlib.pyplot as plt
from collections import defaultdict

######################################################################
# User settings
######################################################################

dataset = 'Wagner_Ring_allperf'   # 'Schubert_Winterreise', 'Wagner_Ring_allperf

do_find_version_pairs = True
do_calculate_warping_paths = True
do_find_two_version_hard_agreement = True
do_find_two_version_soft_agreement = False
do_merge_agreed_frames = False   # do this to merge agreed frames of all pairs of versions for each piece (for hard and soft agreement)

do_calculate_two_version_average_labels = True

######################################################################
# Define dataset splits (cross-version neither split)
######################################################################

if dataset == 'Schubert_Winterreise':
    pass

elif dataset == 'Wagner_Ring_allperf':

    train_songs = ['WWV086B-2', 'WWV086C-2', 'WWV086D-1', 'WWV086B-3', 'WWV086D-3', 'WWV086C-1', 'WWV086D-2']
    val_songs = ['WWV086C-3', 'WWV086B-1']
    test_songs = ['WWV086A', 'WWV086D-0']
    train_versions = ['MEMBRAN2013', 'DG2012', 'PHILIPS2006', 'EMI2012', 'DECCA2012', 'DG2013', 'DECCA2008', 'OEHMS2013', 'NAXOS2003']
    val_versions = ['PROFIL2013', 'SONY2012', 'MEMBRAN1995']
    test_versions = ['ZYX2012', 'EMI2011', 'ORFEO2010']
    

######################################################################
# Find version pairs of the dataset (same key, different version)
######################################################################
if do_find_version_pairs:

    if dataset == 'Schubert_Winterreise':
        ann_audio_global_key = pd.read_csv(os.path.join('/mnt/d/Datasets/original_datasets', dataset, 'ann_audio_globalkey.csv'), sep=';')

        pairs = []

        all_workIDs = ann_audio_global_key['WorkID'].unique()

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
                            'fn_aligned': '_'.join([workID, versions[i], versions[j]])+'.npy',
                        }
                        print(pair)

                        pairs.append(pair)

        print('Found %d pairs of versions' % len(pairs))

    if dataset == 'Wagner_Ring_allperf':

        def get_fn_by_song_version(song, version):
            for fn in os.listdir('/mnt/d/Datasets/precomputed_features/Wagner_Ring_allperf/hcqt_hs512_o6_h5_s1'):
                if song in fn and version in fn:
                    return fn

        pairs = []

        def update_pairs_by_split(songs, versions):
            for song in songs:
                for i in range(len(versions)-1):
                    for j in range(i+1, len(versions)):
                        pair = {
                            'WorkID': song,
                            'PerformanceID1': versions[i],
                            'PerformanceID2': versions[j],
                            'fn1': get_fn_by_song_version(song, versions[i]),
                            'fn2': get_fn_by_song_version(song, versions[j]),
                            'fn_aligned': '_'.join([song, versions[i], versions[j]]) + '.npy',
                        }
                        print(pair)

                        pairs.append(pair)
        
        update_pairs_by_split(train_songs, train_versions)
        update_pairs_by_split(val_songs, val_versions)

        print('Found %d pairs of versions' % len(pairs))

    # Save pairs to csv
    df_pairs = pd.DataFrame(pairs)
    df_pairs.to_csv(os.path.join('/home/lel79bc/workspace/cross-version-mpe/feature_preparation', dataset, 'version_pairs.csv'), index=False)

######################################################################
# Calculate warping paths for each pair of versions
######################################################################

warping_path = os.path.join('/home/lel79bc/workspace/cross-version-mpe/feature_preparation', dataset, 'warping_path')
if not os.path.exists(warping_path):
    os.makedirs(warping_path)

if do_calculate_warping_paths:
    hcqt_path = os.path.join('/mnt/d/Datasets/precomputed_features', dataset, 'hcqt_hs512_o6_h5_s1/')

    df_pairs = pd.read_csv(os.path.join('/home/lel79bc/workspace/cross-version-mpe/feature_preparation', dataset, 'version_pairs.csv'))

    for i, pair in df_pairs.iterrows():
        print('Processing pair %d of %d' % (i+1, len(df_pairs)))

        fn1 = pair['fn1']
        fn2 = pair['fn2']
        fn_aligned = pair['fn_aligned']

        # load hcqt features for both versions
        f_hcqt1 = np.load(os.path.join(hcqt_path, fn1))
        f_hcqt2 = np.load(os.path.join(hcqt_path, fn2))

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

        # # plot and investigate warping path
        # plt.figure()
        # plt.scatter(alignment[0, :], alignment[1, :], s=1)
        # plt.xlabel('Version 1')
        # plt.ylabel('Version 2')
        # plt.savefig('output.png')
        # input('Press Enter to continue...')
        
        np.save(os.path.join(warping_path, fn_aligned), alignment)
    
######################################################################
# Calculate cross-version agreement for each pair of versions (two versions hard agreement)
######################################################################

if do_find_two_version_hard_agreement:

    # Agreement settings
    eval_thresh = 0.5  # binary threshold for teacher annotations
    pitched_only = True   # only consider pitched frames
    allow_diff = 2  # allow at most 1 pixel difference per frame (for calculating agreement)
    align_tolerance = 2  # allow +-N frames difference for soft alignment (for calculating agreement), initially set to 0

    # Paths
    teacher_annotations_path = os.path.join('/home/lel79bc/workspace/cross-version-mpe/feature_preparation', dataset, 'teacher_annotations')

    two_version_agreed_frames_path = os.path.join('/home/lel79bc/workspace/cross-version-mpe/feature_preparation', dataset, 'two_version_agreed_frames.eval_thresh={:.2f}.pitched_only={}.allow_diff={}.align_tolerance={}'.format(eval_thresh, pitched_only, allow_diff, align_tolerance))
    if not os.path.exists(two_version_agreed_frames_path):
        os.makedirs(two_version_agreed_frames_path)

    df_pairs = pd.read_csv(os.path.join(os.path.join('/home/lel79bc/workspace/cross-version-mpe/feature_preparation', dataset, 'version_pairs.csv')))

    percentages_agreed = []
    percentages_agreed_pitched = []
    percentages_pitched_over_all = []
    percentages_pitched_over_agreed = []

    for i, pair in df_pairs.iterrows():
        print('Processing pair %d of %d' % (i+1, len(df_pairs)), end='\r')

        # Load alignment warping path
        fn_aligned = pair['fn_aligned']
        alignment = np.load(os.path.join(warping_path, fn_aligned)).astype(int)   # (2, n_frames)
        
        # Get teacher annotations for both versions
        teacher_annotations_raw_1 = np.load(os.path.join(teacher_annotations_path, pair['fn1']))   # (n_frames, 72)
        teacher_annotations_raw_2 = np.load(os.path.join(teacher_annotations_path, pair['fn2']))   # (n_frames, 72)
        teacher_annotations_1 = (teacher_annotations_raw_1 > eval_thresh).astype(int)
        teacher_annotations_2 = (teacher_annotations_raw_2 > eval_thresh).astype(int)
        assert alignment[0, -1] + 1 == teacher_annotations_1.shape[0]
        assert alignment[1, -1] + 1 == teacher_annotations_2.shape[0]

        # Check agreement of teacher annotations (binary) between the two versions
        agreed_index_1, agreed_index_2 = set(), set()
        for index_1, index_2 in zip(alignment[0, :], alignment[1, :]):
            if np.sum((teacher_annotations_1[index_1,:] == teacher_annotations_2[index_2,:]) == 0) <= allow_diff:
                agreed_index_1.add(index_1)
                agreed_index_2.add(index_2)
            else:
                # soft alignment
                if align_tolerance > 0:
                    # check version 1 agreement
                    agreed = False
                    for idx_2 in range(index_2 - align_tolerance, index_2 + align_tolerance+1):
                        if idx_2 > 0 and idx_2 < teacher_annotations_2.shape[0] and idx_2 != index_2:
                            if np.sum((teacher_annotations_1[index_1,:] == teacher_annotations_2[idx_2,:]) == 0) <= allow_diff:
                                agreed = True
                                break
                    if agreed:
                        agreed_index_1.add(index_1)
                    # check version 2 agreement
                    agreed = False
                    for idx_1 in range(index_1 - align_tolerance, index_1 + align_tolerance+1):
                        if idx_1 > 0 and idx_1 < teacher_annotations_1.shape[0] and idx_1 != index_1:
                            if np.sum((teacher_annotations_1[idx_1,:] == teacher_annotations_2[index_2,:]) == 0) <= allow_diff:
                                agreed = True
                                break
                    if agreed:
                        agreed_index_2.add(index_2)

        # Calculate agreed percentage
        def calculate_agreed_percentage(agreed_index, teacher_annotations):
            percentage_agreed = len(agreed_index) / teacher_annotations.shape[0]
            
            count_pitched_frames = 0
            count_pitched_frames_agreed = 0
            for idx in range(teacher_annotations.shape[0]):
                if teacher_annotations[idx, :].sum() == 0:
                    continue
                count_pitched_frames += 1
                if idx in agreed_index:
                    count_pitched_frames_agreed += 1
            percentage_agreed_pitched = count_pitched_frames_agreed / teacher_annotations.shape[0]
            percentage_pitched_over_all = count_pitched_frames / teacher_annotations.shape[0]
            percentage_pitched_over_agreed = count_pitched_frames_agreed / len(agreed_index)
            
            percentages_agreed.append(percentage_agreed)
            percentages_agreed_pitched.append(percentage_agreed_pitched)
            percentages_pitched_over_all.append(percentage_pitched_over_all)
            percentages_pitched_over_agreed.append(percentage_pitched_over_agreed)

        calculate_agreed_percentage(agreed_index_1, teacher_annotations_1)
        calculate_agreed_percentage(agreed_index_2, teacher_annotations_2)

        # Save the agreed index to file
        fn1 = os.path.join(two_version_agreed_frames_path, '_agree_by_'.join([pair['fn1'], pair['fn_aligned']]))
        fn2 = os.path.join(two_version_agreed_frames_path, '_agree_by_'.join([pair['fn2'], pair['fn_aligned']]))
        # Remove non-pitched frames if needed
        if pitched_only:
            agreed_index_1_pitched = []
            agreed_index_2_pitched = []
            for idx in agreed_index_1:
                if teacher_annotations_1[idx, :].sum() > 0:
                    agreed_index_1_pitched.append(idx)
            for idx in agreed_index_2:
                if teacher_annotations_2[idx, :].sum() > 0:
                    agreed_index_2_pitched.append(idx)
            agreed_index_1 = set(agreed_index_1_pitched)
            agreed_index_2 = set(agreed_index_2_pitched)
        # Save to file
        np.save(fn1, np.array(list(agreed_index_1)))
        np.save(fn2, np.array(list(agreed_index_2)))
    
    print()

    print('='*20)
    print('Mean percentage agreed: {:.4f}'.format(np.mean(percentages_agreed)))
    print('Mean percentage agreed pitched: {:.4f}'.format(np.mean(percentages_agreed_pitched)))
    print('Mean percentage pitched over all: {:.4f}'.format(np.mean(percentages_pitched_over_all)))
    print('Mean percentage pitched over agreed: {:.4f}'.format(np.mean(percentages_pitched_over_agreed)))


######################################################################
# Calculate cross-version agreement for each pair of versions (two versions soft agreement)
######################################################################
if do_find_two_version_soft_agreement:
    print('FINDING SOFT AGREEMENT')

    # Agreement settings
    agreement_ratio = 0.5  # ratio of frames that we want to agree on, initially set to 0.5
    pitched_only = True   # only consider pitched frames, initially set to True
    eval_thresh = 0.4    # binary threshold for teacher annotations (for pitched frames), initially set to 0.4
    agreement_metric = 'cosine'  # 'euclidean' or 'cosine', initially set to 'euclidean', !!! Convert all metric to the smaller the better !!!
    align_tolerance = 0    # allow +-N frames difference for soft alignment (for calculating agreement), initially set to 0
    
    # Paths
    teacher_annotations_path = os.path.join('/home/lel79bc/workspace/cross-version-mpe/feature_preparation', dataset, 'teacher_annotations')
    two_version_soft_agreed_frames_path = os.path.join('/home/lel79bc/workspace/cross-version-mpe/feature_preparation', dataset, 'two_version_soft_agreed_frames.agreement_ratio={:.2f}.pitched_only={}.eval_thresh={:.2f}.agreement_metric={}.align_tolerance={}'.format(agreement_ratio, pitched_only, eval_thresh, agreement_metric, align_tolerance))
    if not os.path.exists(two_version_soft_agreed_frames_path):
        os.makedirs(two_version_soft_agreed_frames_path)
    
    df_pairs = pd.read_csv(os.path.join(os.path.join('/home/lel79bc/workspace/cross-version-mpe/feature_preparation', dataset, 'version_pairs.csv')))

    print('Calculating agreement metric...')

    metrics_all = []  # metrics between teacher annotations of two versions
    metrics_per_pair = []  # save the framewise metrics for each pair of versions

    for i, pair in df_pairs.iterrows():
        print('Processing pair %d of %d' % (i+1, len(df_pairs)), end='\r')

        # Load alignment warping path
        fn_aligned = pair['fn_aligned']
        alignment = np.load(os.path.join(warping_path, fn_aligned)).astype(int)   # (2, n_frames)

        # Get teacher annotations for both versions
        teacher_annotations_raw_1 = np.load(os.path.join(teacher_annotations_path, pair['fn1']))  # (n_frames, 72)
        teacher_annotations_raw_2 = np.load(os.path.join(teacher_annotations_path, pair['fn2']))  # (n_frames, 72)

        # Loop through all candidate alignment frames
        metrics_per_pair.append([])
        for j in range(alignment.shape[1]):
            index_1 = alignment[0, j]
            index_2 = alignment[1, j]

            # Get metric for performance 1, index_1
            teacher_annotation_1 = teacher_annotations_raw_1[index_1, :]
            metric = np.inf
            for idx_2 in range(max(0, index_2 - align_tolerance), min(teacher_annotations_raw_2.shape[0], index_2 + align_tolerance + 1)):
                teacher_annotation_2 = teacher_annotations_raw_2[idx_2, :]

                # Calculate agreement metric between the two versions
                if agreement_metric == 'euclidean':
                    metric_cur = np.linalg.norm(teacher_annotation_1 - teacher_annotation_2)
                elif agreement_metric == 'cosine':
                    metric_cur = np.dot(teacher_annotation_1, teacher_annotation_2) / (np.linalg.norm(teacher_annotation_1) * np.linalg.norm(teacher_annotation_2))
                    metric_cur = 1 - metric_cur  # convert to the smaller the better
                
                if metric_cur < metric:
                    metric = metric_cur
            
            metrics_all.append(metric)
            metrics_per_pair[-1].append(metric)

            # Get metric for performance 2, index_2
            teacher_annotation_2 = teacher_annotations_raw_2[index_2, :]
            metric = np.inf
            for idx_1 in range(max(0, index_1 - align_tolerance), min(teacher_annotations_raw_1.shape[0], index_1 + align_tolerance + 1)):
                teacher_annotation_1 = teacher_annotations_raw_1[idx_1, :]

                # Calculate agreement metric between the two versions
                if agreement_metric == 'euclidean':
                    metric_cur = np.linalg.norm(teacher_annotation_1 - teacher_annotation_2)
                elif agreement_metric == 'cosine':
                    metric_cur = np.dot(teacher_annotation_1, teacher_annotation_2) / (np.linalg.norm(teacher_annotation_1) * np.linalg.norm(teacher_annotation_2))
                    metric_cur = 1 - metric_cur

                if metric_cur < metric:
                    metric = metric_cur

            metrics_all.append(metric)
            metrics_per_pair[-1].append(metric)

            # # Get teacher annotations for the candidate alignment frame
            # teacher_annotation_1 = teacher_annotations_raw_1[index_1, :]
            # teacher_annotation_2 = teacher_annotations_raw_2[index_2, :]

            # # Calculate the metric between the two versions
            # if agreement_metric == 'euclidean':
            #     metric = np.linalg.norm(teacher_annotation_1 - teacher_annotation_2)
            # elif agreement_metric == 'cosine':
            #     metric = np.dot(teacher_annotation_1, teacher_annotation_2) / (np.linalg.norm(teacher_annotation_1) * np.linalg.norm(teacher_annotation_2))
            #     metric = 1 - metric  # convert to the smaller the better

            # # Update the agreement metric
            # metrics_all.append(metric)
            # metrics_per_pair[-1].append(metric)

    print()

    # # Plot the distribution of euclidean distances
    # print('Plot distribution of agreement metric...')
    # if not os.path.exists('figures'):
    #     os.makedirs('figures')
    # plt.figure()
    # plt.hist(metrics_all, bins=100)
    # plt.ylabel('Count')

    # if agreement_metric == 'euclidean':
    #     plt.xlabel('Euclidean distance')
    #     plt.savefig('figures/soft_agreement_distribution_euclidean_distance.align_tolerance={}.png'.format(align_tolerance))
    # elif agreement_metric == 'cosine':
    #     plt.xlabel('1 - Cosine similarity')
    #     plt.savefig('figures/soft_agreement_distribution_cosine_similarity.align_tolerance={}.png'.format(align_tolerance))

    # Calculate the cutoff threshold for euclidean distance by agreement ratio
    metrics_all = np.array(metrics_all)
    metrics_all.sort()
    cutoff_threshold = metrics_all[int(len(metrics_all) * agreement_ratio)]
    print('Cutoff threshold: {:.4f}'.format(cutoff_threshold))

    # Save the agreed index to file
    print('Saving agreed frames to file...')

    n_frames_all = len(metrics_all) * 2  # two versions for each pair
    n_agreed_pitched_framesa_all = 0

    for i, pair in df_pairs.iterrows():
        print('Processing pair %d of %d' % (i+1, len(df_pairs)), end='\r')

        # Load alignment warping path
        fn_aligned = pair['fn_aligned']
        alignment = np.load(os.path.join(warping_path, fn_aligned)).astype(int)   # (2, n_frames)

        agreed_index_1, agreed_index_2 = set(), set()

        # Loop through all candidate alignment frames
        for j in range(alignment.shape[1]):
            index_1 = alignment[0, j]
            index_2 = alignment[1, j]

            # Get the euclidean distance between the two versions
            # metric = metrics_per_pair[i][j]
            metric_1 = metrics_per_pair[i][j*2]
            metric_2 = metrics_per_pair[i][j*2+1]

            # If the euclidean distance is smaller than the cutoff threshold, save the frame
            # if metric < cutoff_threshold:
            #     agreed_index_1.add(index_1)
            #     agreed_index_2.add(index_2)
            if metric_1 < cutoff_threshold:
                agreed_index_1.add(index_1)
            if metric_2 < cutoff_threshold:
                agreed_index_2.add(index_2)

        # Get pitched frames
        agreed_index_1_pitched, agreed_index_2_pitched = set(), set()
        
        teacher_annotations_raw_1 = np.load(os.path.join(teacher_annotations_path, pair['fn1']))  # (n_frames, 72)
        teacher_annotations_raw_2 = np.load(os.path.join(teacher_annotations_path, pair['fn2']))  # (n_frames, 72)
        teacher_annotations_1 = (teacher_annotations_raw_1 > eval_thresh).astype(int)
        teacher_annotations_2 = (teacher_annotations_raw_2 > eval_thresh).astype(int)

        for idx in agreed_index_1:
            if teacher_annotations_1[idx, :].sum() > 0:
                agreed_index_1_pitched.add(idx)
        for idx in agreed_index_2:
            if teacher_annotations_2[idx, :].sum() > 0:
                agreed_index_2_pitched.add(idx)

        n_agreed_pitched_framesa_all += len(agreed_index_1_pitched) + len(agreed_index_2_pitched)
                
        # Save to file
        fn1 = os.path.join(two_version_soft_agreed_frames_path, '_agree_by_'.join([pair['fn1'], pair['fn_aligned']]))
        fn2 = os.path.join(two_version_soft_agreed_frames_path, '_agree_by_'.join([pair['fn2'], pair['fn_aligned']]))
        if pitched_only:
            np.save(fn1, np.array(list(agreed_index_1_pitched)))
            np.save(fn2, np.array(list(agreed_index_2_pitched)))
        else:
            np.save(fn1, np.array(list(agreed_index_1)))
            np.save(fn2, np.array(list(agreed_index_2)))

    print()

    print('====================================================')
    print('Agreement ratio: {:.2f}'.format(agreement_ratio))
    print('Agreed & Pitched ratio: {:.2f}'.format(n_agreed_pitched_framesa_all / n_frames_all))
    

######################################################################
# Merge agreed frames of all pairs of versions for each piece
######################################################################
if do_merge_agreed_frames:

    agreed_frames_folder = 'two_version_soft_agreed_frames.agreement_ratio=0.50.pitched_only=True.eval_thresh=0.40.agreement_metric=cosine.align_tolerance=1'

    two_version_agreed_frames_path = os.path.join('/home/lel79bc/workspace/cross-version-mpe/feature_preparation', dataset, agreed_frames_folder)
    df_pairs = pd.read_csv(os.path.join(os.path.join('/home/lel79bc/workspace/cross-version-mpe/feature_preparation', dataset, 'version_pairs.csv')))

    agreed_frames_by_fn = defaultdict(set)  # fn to set of agreed frames

    # Merge agreed frames for each piece
    for i, pair in df_pairs.iterrows():
        print('Processing pair %d of %d' % (i+1, len(df_pairs)), end='\r')

        # Get filenames
        fn1 = pair['fn1']
        fn2 = pair['fn2']

        # Get the agreed frames for both versions
        fn1_agreed_frames = os.path.join(two_version_agreed_frames_path, '_agree_by_'.join([pair['fn1'], pair['fn_aligned']]))
        fn2_agreed_frames = os.path.join(two_version_agreed_frames_path, '_agree_by_'.join([pair['fn2'], pair['fn_aligned']]))
        agreed_frames_1 = np.load(fn1_agreed_frames)
        agreed_frames_2 = np.load(fn2_agreed_frames)

        # Merge the agreed frames
        agreed_frames_by_fn[fn1].update(agreed_frames_1)
        agreed_frames_by_fn[fn2].update(agreed_frames_2)

    # Save the agreed frames to file
    print('Saving agreed frames to file...')
    for fn in agreed_frames_by_fn:
        # sort agreed frames
        agreed_frames = np.array(list(agreed_frames_by_fn[fn]))
        agreed_frames.sort()
        # save to file
        np.save(os.path.join(two_version_agreed_frames_path, fn), agreed_frames)


######################################################################
# Calculate two-version average labels
######################################################################
        
if do_calculate_two_version_average_labels:

    # Agreement settings
    eval_thresh = 0.5  # binary threshold for teacher annotations

    # Paths
    teacher_annotations_path = os.path.join('/home/lel79bc/workspace/cross-version-mpe/feature_preparation', dataset, 'teacher_annotations')

    two_version_average_labels_path = os.path.join('/home/lel79bc/workspace/cross-version-mpe/feature_preparation', dataset, 'two_version_average_labels.eval_thresh={:.2f}'.format(eval_thresh))
    if not os.path.exists(two_version_average_labels_path):
        os.makedirs(two_version_average_labels_path)

    df_pairs = pd.read_csv(os.path.join(os.path.join('/home/lel79bc/workspace/cross-version-mpe/feature_preparation', dataset, 'version_pairs.csv')))

    count_pixels_positive, count_pixels_negative, count_pixels_half = 0, 0, 0
    count_pixels_positive_pitched, count_pixels_negative_pitched, count_pixels_half_pitched = 0, 0, 0

    for i, pair in df_pairs.iterrows():
        print('Processing pair %d of %d' % (i+1, len(df_pairs)), end='\r')

        # Load alignment warping path
        fn_aligned = pair['fn_aligned']
        alignment = np.load(os.path.join(warping_path, fn_aligned)).astype(int)   # (2, n_frames)

        # Get teacher annotations for both versions
        teacher_annotations_raw_1 = np.load(os.path.join(teacher_annotations_path, pair['fn1']))   # (n_frames, 72)
        teacher_annotations_raw_2 = np.load(os.path.join(teacher_annotations_path, pair['fn2']))   # (n_frames, 72)
        teacher_annotations_1 = (teacher_annotations_raw_1 > eval_thresh).astype(int)
        teacher_annotations_2 = (teacher_annotations_raw_2 > eval_thresh).astype(int)
        assert alignment[0, -1] + 1 == teacher_annotations_1.shape[0]
        assert alignment[1, -1] + 1 == teacher_annotations_2.shape[0]

        # Calculate the average labels
        teacher_annotations_1_averaged = np.zeros(teacher_annotations_1.shape, dtype=float)
        teacher_annotations_2_averaged = np.zeros(teacher_annotations_2.shape, dtype=float)

        for index_1, index_2 in zip(alignment[0, :], alignment[1, :]):
            ave_label = (teacher_annotations_1[index_1,:] + teacher_annotations_2[index_2,:]) / 2
            teacher_annotations_1_averaged[index_1,:] = ave_label
            teacher_annotations_2_averaged[index_2,:] = ave_label

        # # Save the average labels to file
        # fn1 = os.path.join(two_version_average_labels_path, '_agree_by_'.join([pair['fn1'], pair['fn_aligned']]))
        # fn2 = os.path.join(two_version_average_labels_path, '_agree_by_'.join([pair['fn2'], pair['fn_aligned']]))
        # np.save(fn1, teacher_annotations_1_averaged)
        # np.save(fn2, teacher_annotations_2_averaged)

        # Calculate the number of positive, negative, and half pixels
        count_pixels_positive += np.sum(teacher_annotations_1_averaged == 1) + np.sum(teacher_annotations_2_averaged == 1)
        count_pixels_negative += np.sum(teacher_annotations_1_averaged == 0) + np.sum(teacher_annotations_2_averaged == 0)
        count_pixels_half += np.sum(teacher_annotations_1_averaged == 0.5) + np.sum(teacher_annotations_2_averaged == 0.5)

        # Calculate the number of positive, negative, and half pixels (pitched_only)
        # Get pitched frames (only positive)
        pitched_frames_idx_1 = np.where((teacher_annotations_1_averaged == 1).sum(axis=1) > 0)[0]
        pitched_frames_idx_2 = np.where((teacher_annotations_2_averaged == 1).sum(axis=1) > 0)[0]

        count_pixels_positive_pitched += np.sum(teacher_annotations_1_averaged[pitched_frames_idx_1,:] == 1) + np.sum(teacher_annotations_2_averaged[pitched_frames_idx_2,:] == 1)
        count_pixels_negative_pitched += np.sum(teacher_annotations_1_averaged[pitched_frames_idx_1,:] == 0) + np.sum(teacher_annotations_2_averaged[pitched_frames_idx_2,:] == 0)
        count_pixels_half_pitched += np.sum(teacher_annotations_1_averaged[pitched_frames_idx_1,:] == 0.5) + np.sum(teacher_annotations_2_averaged[pitched_frames_idx_2,:] == 0.5)


    print()

    # Calculate the percentage of positive, negative, and half pixels
    total_pixels = count_pixels_positive + count_pixels_negative + count_pixels_half
    percentage_pixels_positive = count_pixels_positive / total_pixels
    percentage_pixels_negative = count_pixels_negative / total_pixels
    percentage_pixels_half = count_pixels_half / total_pixels

    total_pixels_pitched = count_pixels_positive_pitched + count_pixels_negative_pitched + count_pixels_half_pitched
    percentage_pixels_positive_pitched = count_pixels_positive_pitched / total_pixels_pitched
    percentage_pixels_negative_pitched = count_pixels_negative_pitched / total_pixels_pitched
    percentage_pixels_half_pitched = count_pixels_half_pitched / total_pixels_pitched

    print('Percentage of positive pixels: {:.4f}'.format(percentage_pixels_positive))
    print('Percentage of negative pixels: {:.4f}'.format(percentage_pixels_negative))
    print('Percentage of half pixels: {:.4f}'.format(percentage_pixels_half))
    print('----------------------------------------------------')
    print('Percentage of positive pixels (pitched_only): {:.4f}'.format(percentage_pixels_positive_pitched))
    print('Percentage of negative pixels (pitched_only): {:.4f}'.format(percentage_pixels_negative_pitched))
    print('Percentage of half pixels (pitched_only): {:.4f}'.format(percentage_pixels_half_pitched))




