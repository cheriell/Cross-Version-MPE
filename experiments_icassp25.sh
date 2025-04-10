# Detailed experiment pipeline for the ICASSP 2025 paper.

# This script also include some repeating steps from the ICLR 2024 tiny paper, such as feature preparation set up, teacher model training, and T, TS, TSCV1, TSCV2 experiments on the Schubert dataset. You can safely skip them if you have already run them.


#############################################################
# Define paths to the data, precomputed features, and workspace.
#############################################################

# Environment variables - Set before running the script
export DATADIR=/home/username/datasets
export CODEDIR=/home/username/repositories/cross-version-mpe
export WORKSPACEDIR=/home/username/workspace/cross-version-mpe
export WANDB_API_KEY=YOUR_API_KEY

# Original datasets
# We use the original dataset paths to get the MAESTRO train/valid/test splits and the Schubert key annotations.
original_dataset_path_maestro=/mnt/d/Datasets/original_datasets/MAESTRO
original_dataset_path_schubert=/mnt/d/Datasets/original_datasets/Schubert_Winterreise

# Precomputed features
# This is the paths where the precomputed features (HCQT and pianorolls) are saved.
precomputed_features_path_maestro=/mnt/d/Datasets/precomputed_features/MAESTRO
precomputed_features_path_schubert=/mnt/d/Datasets/precomputed_features/Schubert_Winterreise
precomputed_features_path_wagner=/mnt/d/Datasets/precomputed_features/Wagner_Ring_allperf
pitch_folder=pitch_hs512_nooverl
pitch_folder_wagner_test=pitch_hs512_nooverl_fullscore
hcqt_folder=hcqt_hs512_o6_h5_s1
hcqt_folder_wagner_test=hcqt_hs512_o6_h5_s1_fullscore

# Segmented precomputed features
# We segment the precomputed features into shorter ones for faster online data loading in the training process.
segmented_features_path_maestro=$DATADIR/segmented_features/MAESTRO
segmented_features_path_schubert=$DATADIR/segmented_features/Schubert_Winterreise
segmented_features_path_wagner=$DATADIR/segmented_features/Wagner_Ring_allperf
segment_length=30  # in seconds

# workspace
# This is where to save the experimental outputs.
workspace=$WORKSPACEDIR


#############################################################
# Feature preparation setup
#############################################################

# Split datasets into train, validation, and test sets.
python3 experiments_icassp25/feature_preparation/00_setup_01_split_datasets.py \
    --original_dataset_path_maestro $original_dataset_path_maestro \
    --precomputed_features_path_maestro $precomputed_features_path_maestro \
    --precomputed_features_path_schubert $precomputed_features_path_schubert \
    --precomputed_features_path_wagner $precomputed_features_path_wagner \
    --pitch_folder $pitch_folder \
    --pitch_folder_wagner_test $pitch_folder_wagner_test

# Segment the precomputed features (pitch and hcqt) into shorter ones.
# This is because the songs in the Wagner dataset is too long, which is not efficient for online loading in the training process.
python3 experiments_icassp25/feature_preparation/00_setup_02_segment_precomputed_features.py \
    --precomputed_features_path_maestro $precomputed_features_path_maestro \
    --precomputed_features_path_schubert $precomputed_features_path_schubert \
    --precomputed_features_path_wagner $precomputed_features_path_wagner \
    --segmented_features_path_maestro $segmented_features_path_maestro \
    --segmented_features_path_schubert $segmented_features_path_schubert \
    --segmented_features_path_wagner $segmented_features_path_wagner \
    --pitch_folder $pitch_folder \
    --hcqt_folder $hcqt_folder \
    --pitch_folder_wagner_test $pitch_folder_wagner_test \
    --hcqt_folder_wagner_test $hcqt_folder_wagner_test \
    --segment_length $segment_length

# Find version pairs for the training and validation sets.
python3 experiments_icassp25/feature_preparation/00_setup_03_find_version_pairs.py \
    --original_dataset_path_schubert $original_dataset_path_schubert \
    --workspace $workspace

# Calculate warping paths for the training and validation sets.
python3 experiments_icassp25/feature_preparation/00_setup_04_calculate_warping_paths.py \
    --workspace $workspace \
    --precomputed_features_path_schubert $precomputed_features_path_schubert \
    --precomputed_features_path_wagner $precomputed_features_path_wagner \
    --hcqt_folder $hcqt_folder


#############################################################
# Experiments: Sup
# Supervised training using the ResCNN model.
#############################################################

# Train the teacher model on MAESTRO datset.
python3 experiments_icassp25/main/run_rescnn.py \
    --runname run0 \
    --config_json_file experiments_icassp25/configs/ts/maestro_sup.json \
    --workspace $workspace \
    --segmented_features_path_maestro $segmented_features_path_maestro \
    --hcqt_folder $hcqt_folder \
    --pitch_folder $pitch_folder

# Supervised training on the Schubert and Wagner dataset.
for config_fn in "schubert_sup.json" "wagner_sup.json"
do
    for runname in "run0" "run1" "run2"
    do
        python3 experiments_icassp25/main/run_rescnn.py \
            --runname $runname \
            --config_json_file experiments_icassp25/configs/ts/$config_fn \
            --workspace $workspace \
            --segmented_features_path_schubert $segmented_features_path_schubert \
            --segmented_features_path_wagner $segmented_features_path_wagner \
            --hcqt_folder $hcqt_folder \
            --pitch_folder $pitch_folder \
            --hcqt_folder_wagner_test $hcqt_folder_wagner_test \
            --pitch_folder_wagner_test $pitch_folder_wagner_test
    done
done


#############################################################
# Experiments: T
# Evaluate the teacher model on the Schubert and Wagner dataset.
#############################################################

# Evaluate the teacher model on the Schubert and Wagner dataset.
for config_fn in "schubert_t.json" "wagner_t.json"
do
    python3 experiments_icassp25/main/run_rescnn.py \
        --runname run0 \
        --config_json_file experiments_icassp25/configs/ts/$config_fn \
        --workspace $workspace \
        --segmented_features_path_schubert $segmented_features_path_schubert \
        --segmented_features_path_wagner $segmented_features_path_wagner \
        --hcqt_folder $hcqt_folder \
        --pitch_folder $pitch_folder \
        --hcqt_folder_wagner_test $hcqt_folder_wagner_test \
        --pitch_folder_wagner_test $pitch_folder_wagner_test \
        --eval_only
done


#############################################################
# Experiments: TS
# Teacher-student training using the ResCNN model.
#############################################################

# Get and segment teacher annotations (pseudo labels).
python3 experiments_icassp25/feature_preparation/01_ts_01_get_rescnn_pseudo_labels.py \
    --dataset schubert \
    --teacher_model_path $workspace/experiments_icassp25/models/maestro_sup-run0.pt \
    --path_x $segmented_features_path_schubert/$hcqt_folder \
    --path_pseudo_labels $workspace/segmented_pseudo_labels/schubert_rescnn \
    --segment_length $segment_length

python3 experiments_icassp25/feature_preparation/01_ts_01_get_rescnn_pseudo_labels.py \
    --dataset wagner \
    --teacher_model_path $workspace/experiments_icassp25/models/maestro_sup-run0.pt \
    --path_x $segmented_features_path_wagner/$hcqt_folder \
    --path_pseudo_labels $workspace/segmented_pseudo_labels/wagner_rescnn \
    --segment_length $segment_length

# Teacher-student learning.
for config_fn in "schubert_ts.json" "wagner_ts.json"
do
    for runname in "run0" "run1" "run2"
    do
        python3 experiments_icassp25/main/run_rescnn.py \
            --runname $runname \
            --config_json_file experiments_icassp25/configs/ts/$config_fn \
            --workspace $workspace \
            --segmented_features_path_schubert $segmented_features_path_schubert \
            --segmented_features_path_wagner $segmented_features_path_wagner \
            --hcqt_folder $hcqt_folder \
            --pitch_folder $pitch_folder \
            --hcqt_folder_wagner_test $hcqt_folder_wagner_test \
            --pitch_folder_wagner_test $pitch_folder_wagner_test \
            --binary_labels
    done
done


#############################################################
# Experiments: TSCV1
# Teacher-student training with cross-version consistency.
#############################################################

## Getting the consistent frames between version pairs for the Schubert and Wagner dataset.

# Hard consistency -- agreement with a maximum of 2 pitch differences per time frame, this is what is presented in the paper.
python3 experiments_icassp25/feature_preparation/01_ts_02_find_consistent_frames.py \
    --dataset schubert \
    --workspace $workspace \
    --path_x $segmented_features_path_schubert/$hcqt_folder \
    --path_pseudo_labels $workspace/segmented_pseudo_labels/schubert_rescnn \
    --path_consistent_frames $workspace/consistent_frames/schubert_rescnn_hard_0.5_2_2 \
    --print_stats \
    hard \
        --threshold 0.5 \
        --allow_diff 2 \
        --align_tolerance 2

python3 experiments_icassp25/feature_preparation/01_ts_02_find_consistent_frames.py \
    --dataset wagner \
    --workspace $workspace \
    --path_x $segmented_features_path_wagner/$hcqt_folder \
    --path_pseudo_labels $workspace/segmented_pseudo_labels/wagner_rescnn \
    --path_consistent_frames $workspace/consistent_frames/wagner_rescnn_hard_0.5_2_2 \
    --print_stats \
    hard \
        --threshold 0.5 \
        --allow_diff 2 \
        --align_tolerance 2

# Soft consistency -- agreement based on a cosine similarity between the predicted pitch probabilities. We didn't include this in the paper since it's not working well.
python3 experiments_icassp25/feature_preparation/01_ts_02_find_consistent_frames.py \
    --dataset schubert \
    --workspace $workspace \
    --path_x $segmented_features_path_schubert/$hcqt_folder \
    --path_pseudo_labels $workspace/segmented_pseudo_labels/schubert_rescnn \
    --path_consistent_frames $workspace/consistent_frames/schubert_rescnn_soft_0.5_0.4_cosine_2 \
    --print_stats \
    soft \
        --cutoff_ratio 0.5 \
        --threshold 0.4 \
        --metric "cosine" \
        --align_tolerance 2

python3 experiments_icassp25/feature_preparation/01_ts_02_find_consistent_frames.py \
    --dataset wagner \
    --workspace $workspace \
    --path_x $segmented_features_path_wagner/$hcqt_folder \
    --path_pseudo_labels $workspace/segmented_pseudo_labels/wagner_rescnn \
    --path_consistent_frames $workspace/consistent_frames/wagner_rescnn_soft_0.5_0.4_cosine_2 \
    --print_stats \
    soft \
        --cutoff_ratio 0.5 \
        --threshold 0.4 \
        --metric "cosine" \
        --align_tolerance 2
    
# Train and evaluate the TSCV1 model on the Schubert and Wagner dataset.
for config_fn in "schubert_tscv.json" "wagner_tscv.json"
do
    for runname in "run0" "run1" "run2"
    do
        python3 experiments_icassp25/main/run_rescnn.py \
            --runname $runname \
            --config_json_file experiments_icassp25/configs/ts/$config_fn \
            --workspace $workspace \
            --segmented_features_path_schubert $segmented_features_path_schubert \
            --segmented_features_path_wagner $segmented_features_path_wagner \
            --hcqt_folder $hcqt_folder \
            --hcqt_folder_wagner_test $hcqt_folder_wagner_test \
            --pitch_folder_wagner_test $pitch_folder_wagner_test \
            --binary_labels
    done
done


#############################################################
# Experiments: TSCV2
# Teacher-student training with cross-version averaged pseudo labels.
#############################################################

# Calculate the averaged pseudo labels for the Schubert and Wagner dataset.
python3 experiments_icassp25/feature_preparation/01_ts_03_calculate_averaged_pseudo_labels.py \
    --dataset schubert \
    --workspace $workspace \
    --path_x $segmented_features_path_schubert/$hcqt_folder \
    --path_pseudo_labels $workspace/segmented_pseudo_labels/schubert_rescnn \
    --path_pseudo_labels_averaged $workspace/segmented_pseudo_labels_averaged/schubert_rescnn \
    --threshold 0.5 \
    --segment_length $segment_length

python3 experiments_icassp25/feature_preparation/01_ts_03_calculate_averaged_pseudo_labels.py \
    --dataset wagner \
    --workspace $workspace \
    --path_x $segmented_features_path_wagner/$hcqt_folder \
    --path_pseudo_labels $workspace/segmented_pseudo_labels/wagner_rescnn \
    --path_pseudo_labels_averaged $workspace/segmented_pseudo_labels_averaged/wagner_rescnn \
    --path_valid_frames $workspace/segmented_pseudo_labels_averaged_valid_frames/wagner_rescnn \
    --threshold 0.5 \
    --segment_length $segment_length

# Train and evaluate the TSCV2 model on the Schubert and Wagner dataset.
for runname in "run0" "run1" "run2"
do
    python3 experiments_icassp25/main/run_rescnn.py \
        --runname $runname \
        --config_json_file experiments_icassp25/configs/ts/wagner_tscv2.json \
        --workspace $workspace \
        --segmented_features_path_schubert $segmented_features_path_schubert \
        --segmented_features_path_wagner $segmented_features_path_wagner \
        --hcqt_folder $hcqt_folder \
        --hcqt_folder_wagner_test $hcqt_folder_wagner_test \
        --pitch_folder_wagner_test $pitch_folder_wagner_test \
        --pick_pairs
done


#############################################################
# Additional path configurations for the AE related experiments
# (AE, AE+TS, and AE+TSCV2)
#############################################################

# Pretrained models (from AE)
path_pretrained_model_schubert=$workspace/experiments_icassp25/aoe/models/schubert_aoe-run0-lr=5e-5.pt
path_pretrained_model_wagner=$workspace/experiments_icassp25/aoe/models/wagner_aoe-run0.pt

# Pseudo labels (from AE)
path_pseudo_labels_schubert=$workspace/experiments_icassp25/aoe/pseudo_labels_schubert
path_pseudo_labels_wagner=$workspace/experiments_icassp25/aoe/pseudo_labels_wagner

# Averaged pseudo labels (from AE)
path_pseudo_labels_averaged_schubert=$workspace/experiments_icassp25/aoe/pseudo_labels_averaged_schubert
path_pseudo_labels_averaged_wagner=$workspace/experiments_icassp25/aoe/pseudo_labels_averaged_wagner
path_pseudo_labels_averaged_valid_frames_schubert=$workspace/experiments_icassp25/aoe/pseudo_labels_averaged_valid_frames_schubert
path_pseudo_labels_averaged_valid_frames_wagner=$workspace/experiments_icassp25/aoe/pseudo_labels_averaged_valid_frames_wagner


#############################################################
# Experiments: AE
#############################################################

# AE: MAESTRO -> Schubert
python3 experiments_icassp25/main/run_aoecnn.py \
    --runname run1 \
    --config_json_file experiments_icassp25/configs/aoe/schubert_aoe.json \
    --workspace $workspace \
    --segmented_features_path_maestro $segmented_features_path_maestro \
    --segmented_features_path_schubert $segmented_features_path_schubert \
    --hcqt_folder $hcqt_folder \
    --pitch_folder $pitch_folder \
    domain_adaptation

# AE: MAESTRO -> Wagner
python3 experiments_icassp25/main/run_aoecnn.py \
    --runname run2.1.eval \
    --config_json_file experiments_icassp25/configs/aoe/wagner_aoe.json \
    --workspace $workspace \
    --segmented_features_path_maestro $segmented_features_path_maestro \
    --segmented_features_path_wagner $segmented_features_path_wagner \
    --hcqt_folder $hcqt_folder \
    --pitch_folder $pitch_folder \
    --hcqt_folder_wagner_test $hcqt_folder_wagner_test \
    --pitch_folder_wagner_test $pitch_folder_wagner_test \
    --eval_only \
    domain_adaptation


#############################################################
# Experiments: AE+TS
#############################################################

## AE+TS: MAESTRO -> Schubert

# Get pseudo labels from the trained baseline AoE model.
python3 experiments_icassp25/feature_preparation/02_aoe_01_get_aoecnn_pseudo_labels.py \
    --dataset schubert \
    --teacher_model_path $path_pretrained_model_schubert \
    --path_x $segmented_features_path_schubert/$hcqt_folder \
    --path_pseudo_labels $path_pseudo_labels_schubert \
    --segment_length $segment_length

# Train and evaluate the AE+TS model
python3 experiments_icassp25/main/run_aoecnn.py \
    --runname run0 \
    --config_json_file experiments_icassp25/configs/aoe/schubert_aoets.json \
    --workspace $workspace \
    --segmented_features_path_schubert $segmented_features_path_schubert \
    --hcqt_folder $hcqt_folder \
    --pitch_folder $pitch_folder \
    teacher_student \
    --path_pretrained_model $path_pretrained_model_schubert \
    --path_pseudo_labels $path_pseudo_labels_schubert \
    --binary_labels

## AE+TS: MAESTRO -> Wagner

# Get pseudo labels from the trained baseline AE model.
python3 experiments_icassp25/feature_preparation/02_aoe_01_get_aoecnn_pseudo_labels.py \
    --dataset wagner \
    --teacher_model_path $path_pretrained_model_wagner \
    --path_x $segmented_features_path_wagner/$hcqt_folder \
    --path_pseudo_labels $path_pseudo_labels_wagner \
    --segment_length $segment_length

# Train and evalute the AE+TS model.
python3 experiments_icassp25/main/run_aoecnn.py \
    --runname run0 \
    --config_json_file experiments_icassp25/configs/aoe/wagner_aoets.json \
    --workspace $workspace \
    --segmented_features_path_wagner $segmented_features_path_wagner \
    --hcqt_folder $hcqt_folder \
    --hcqt_folder_wagner_test $hcqt_folder_wagner_test \
    --pitch_folder_wagner_test $pitch_folder_wagner_test \
    teacher_student \
    --path_pretrained_model $path_pretrained_model_wagner \
    --path_pseudo_labels $path_pseudo_labels_wagner \
    --binary_labels


#############################################################
# Experiments: AE+TSCV2
#############################################################

## AE+TSCV2: MAESTRO -> Schubert

# Get cross-version averaged pseudo labels from the trained baseline AoE model.
# Here, we borrow the feature preparation script from the TS experiments.
python3 experiments_icassp25/feature_preparation/01_ts_03_calculate_averaged_pseudo_labels.py \
    --dataset schubert \
    --workspace $workspace \
    --path_x $segmented_features_path_schubert/$hcqt_folder \
    --path_pseudo_labels $path_pseudo_labels_schubert \
    --path_pseudo_labels_averaged $path_pseudo_labels_averaged_schubert \
    --path_valid_frames $path_pseudo_labels_averaged_valid_frames_schubert \
    --threshold 0.4 \
    --segment_length $segment_length

# Train and evaluate the AE+TSCV2 model.
python3 experiments_icassp25/main/run_aoecnn.py \
    --runname run0 \
    --config_json_file experiments_icassp25/configs/aoe/schubert_aoetscv.json \
    --workspace $workspace \
    --segmented_features_path_schubert $segmented_features_path_schubert \
    --hcqt_folder $hcqt_folder \
    --pitch_folder $pitch_folder \
    teacher_student \
    --path_pretrained_model $path_pretrained_model_schubert \
    --path_pseudo_labels $path_pseudo_labels_averaged_schubert \
    --path_valid_frames $path_pseudo_labels_averaged_valid_frames_schubert \
    --pick_pairs

## AE+TSCV: MAESTRO -> Wagner

# Get crosss-version averaged pseudo labels from the trained baseline AoE mdoel.
# Here, we borrow the feature preparation script from the TS experiments.
python3 experiments_icassp25/feature_preparation/01_ts_03_calculate_averaged_pseudo_labels.py \
    --dataset wagner \
    --workspace $workspace \
    --path_x $segmented_features_path_wagner/$hcqt_folder \
    --path_pseudo_labels $path_pseudo_labels_wagner \
    --path_pseudo_labels_averaged $path_pseudo_labels_averaged_wagner \
    --path_valid_frames $path_pseudo_labels_averaged_valid_frames_wagner \
    --threshold 0.4 \
    --segment_length $segment_length

# Train and evaluate the AE+TSCV2 model.
python3 experiments_icassp25/main/run_aoecnn.py \
    --runname run0 \
    --config_json_file experiments_icassp25/configs/aoe/wagner_aoetscv.json \
    --workspace $workspace \
    --segmented_features_path_wagner $segmented_features_path_wagner \
    --hcqt_folder $hcqt_folder \
    --hcqt_folder_wagner_test $hcqt_folder_wagner_test \
    --pitch_folder_wagner_test $pitch_folder_wagner_test \
    teacher_student \
    --path_pseudo_labels $path_pseudo_labels_averaged_wagner \
    --path_valid_frames $path_pseudo_labels_averaged_valid_frames_wagner \
    --pick_pairs





