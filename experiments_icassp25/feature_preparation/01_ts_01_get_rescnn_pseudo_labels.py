import warnings
warnings.filterwarnings("ignore")

import argparse
import torch
import json
import numpy as np
import os, sys
sys.path.insert(0, os.path.join(sys.path[0], '../..'))

from experiments_icassp25.models.basic_cnns import deep_cnn_segm_sigmoid
from experiments_icassp25.feature_preparation.utils import segment_feature, get_feature_segment_fn, get_feature_segment_metainfo_fn


model_params = {
    'n_chan_input': 6,
    'n_chan_layers': [70,70,50,10],
    'n_prefilt_layers': 5,
    'residual': True,
    'n_bins_in': 216,
    'n_bins_out': 72,
    'a_lrelu': 0.3,
    'p_dropout': 0.2
}
half_context = 75 // 2
device = 'cuda:0'

def main(args):

    # Create the directory for the pseudo labels
    os.makedirs(args.path_pseudo_labels, exist_ok=True)

    # Get the teacher model
    model = deep_cnn_segm_sigmoid(
        n_chan_input=model_params['n_chan_input'],
        n_chan_layers=model_params['n_chan_layers'],
        n_prefilt_layers=model_params['n_prefilt_layers'],
        residual=model_params['residual'],
        n_bins_in=model_params['n_bins_in'],
        n_bins_out=model_params['n_bins_out'],
        a_lrelu=model_params['a_lrelu'],
        p_dropout=model_params['p_dropout'],
    )
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
                print('Segment {} already processed, skip.'.format(seg_idx))
                continue

            # Get input feature for the segment
            x = np.load(os.path.join(args.path_x, get_feature_segment_fn(fn, seg_idx)))  # (n_bins, n_frames, n_chan)
            if seg_idx == 0:
                context_left = np.zeros((x.shape[0], half_context, x.shape[2]))
            else:
                context_left = np.load(os.path.join(args.path_x, get_feature_segment_fn(fn, seg_idx-1)))[:, -half_context:, :]
            if seg_idx == n_segments-1:
                context_right = np.zeros((x.shape[0], half_context, x.shape[2]))
            else:
                context_right = np.load(os.path.join(args.path_x, get_feature_segment_fn(fn, seg_idx+1)))[:, :half_context, :]
                # pad the context_right if it is shorter than half_context
                if context_right.shape[1] < half_context:
                    context_right = np.pad(context_right, ((0, 0), (0, half_context - context_right.shape[1]), (0, 0)), 'constant', constant_values=0)
            x_context = np.concatenate([context_left, x, context_right], axis=1)  # (n_bins, n_frames, n_chan)

            # Get the pseudo labels
            x_context = torch.tensor(x_context).to(device).float()
            x_context = x_context.permute(2, 1, 0).unsqueeze(0)  # (n_batch, n_chan, n_frames, n_bins)
            y_pred = model(x_context)  # (n_batch, n_chan, n_frames, n_bins)
            y_pred = y_pred.squeeze(1).squeeze(0).detach().cpu().numpy()  # (n_frames, n_bins)
            y_pred = y_pred.T  # (n_bins, n_frames)
            
            # Save the pseudo labels
            np.save(os.path.join(args.path_pseudo_labels, get_feature_segment_fn(fn, seg_idx)), y_pred)

        print()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get and save the segmented pseudo labels.')

    parser.add_argument('--dataset', type=str, required=True, help="schubert or wagner")
    parser.add_argument('--teacher_model_path', type=str, required=True, help="Path to the teacher model.")

    parser.add_argument('--path_x', type=str, required=True)
    parser.add_argument('--path_pseudo_labels', type=str, required=True)
    parser.add_argument('--segment_length', type=int, required=True)

    args = parser.parse_args()

    main(args)

