import warnings
warnings.filterwarnings("ignore")

import argparse
import torch
import json
import numpy as np
import os, sys
sys.path.insert(0, os.path.join(sys.path[0], '../..'))

from experiments.models.aoe_cnns import aoe_cnn_model
from experiments.feature_preparation.utils import get_feature_segment_fn, get_feature_segment_metainfo_fn, get_x_context_by_segment_idx


# Model parameters
ae_layers = 3
half_context = 75 // 2
device = 'cuda:0'


def main(args):

    # Create the directory for the pseudo labels
    os.makedirs(args.path_pseudo_labels, exist_ok=True)

    # Get the teacher model
    model = aoe_cnn_model(ae_layers=3)
    model.load_state_dict(torch.load(args.teacher_model_path))
    model.to(device)
    model.eval()

    # Get the dataset splits
    splits = json.load(open('dataset_splits/{}_split.json'.format(args.dataset), 'r'))

    # Get all fns, excluding the testing set
    fns = splits['train'] + splits['val']

    # Process each file
    for fn_idx, fn in enumerate(fns):
        print('Processing file {}/{}'.format(fn_idx+1, len(fns)))

        # Get n_segments
        fn_metainfo = get_feature_segment_metainfo_fn(fn)
        seg_metainfo = json.load(open(os.path.join(args.path_x, fn_metainfo), 'r'))
        n_segments = int(seg_metainfo['n_segments'])

        # Process each segment
        for seg_idx in range(n_segments):
            print('Processing segment {}/{}'.format(seg_idx+1, n_segments), end='\r')

            if os.path.exists(os.path.join(args.path_pseudo_labels, get_feature_segment_fn(fn, seg_idx))):
                continue

            # Get the input feature for the segment
            x_context = get_x_context_by_segment_idx(args.path_x, fn, seg_idx, half_context, n_segments)  # (n_bins, n_frames, n_chan)

            # Get the pseudo labels
            x_context = torch.tensor(x_context).to(device).float()
            x_context = x_context.permute(2, 1, 0).unsqueeze(0)  # (n_batch, n_chan, n_frames, n_bins)
            y_pred = model.predict(x_context)  # (n_batch, n_chan, n_frames, n_bins)
            y_pred = y_pred.squeeze(1).squeeze(0).detach().cpu().numpy()  # (n_frames, n_bins)
            y_pred = y_pred.T  # (n_bins, n_frames)

            # Save the pseudo labels
            np.save(os.path.join(args.path_pseudo_labels, get_feature_segment_fn(fn, seg_idx)), y_pred)

        print()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get and save the segmented pseudo labels.')

    parser.add_argument('--dataset', type=str, help='schubert or wagner')
    parser.add_argument('--teacher_model_path', type=str, help='Path to the teacher model.')

    parser.add_argument('--path_x', type=str, help='Path to the input features.')
    parser.add_argument('--path_pseudo_labels', type=str, help='Path to save the pseudo labels.')
    parser.add_argument('--segment_length', type=int, help='Length of the segment.')

    args = parser.parse_args()

    main(args)
