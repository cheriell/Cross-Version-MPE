# Cross-Version-MPE

Here we provide code for our project of _Domain Adaptation for Music Transcription Using Cross-Version Consistency_. 

Experiments in this repo are presented in the following two papers:

- Lele Liu and Christof Weiß, "[Utilizing cross-version consistency for domain adaptation: A case study on music audio](https://openreview.net/forum?id=ZNg3YQQKWT),” in Tiny Papers Track at International Conference on Learning Representations (Tiny Papers @ ICLR), 2024.
- Lele Liu and Christof Weiß, "[Unsupervised Domain Adaptation for Music Transcription: Exploiting Cross-Version Consistency](https://ieeexplore.ieee.org/document/10889474)," accepted to the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2025.

## Datasets and feature preparation

We use the [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro), the [Schubert Winterreise Dataset](https://zenodo.org/records/3968389) and the [Wagner Ring Dataset](https://zenodo.org/records/7672157) in our case study. 

During feature preparation, we calculate the HCQT spectrogram and binary pianoroll for each of the music performance, with a sampling rate of 22.05 kHz and a hop size of 512.
The feature preparation code can be found [HERE](https://github.com/christofw/multipitch_architectures/blob/master/01_precompute_features.ipynb).
We provide one example input-output pair of feature in the folder `example_precomputed_features`.

## Environments configuration

We use [synctoolbox](https://github.com/meinardmueller/synctoolbox) to calculate the alignment path between different versions. The python environment is the one provided by the toolbox (copied in file `environment-synctoolbox.yml`). Please use this python environment to run the feature preparation script `feature_preparation/prepare_cross_version_alignment.py`.

To run the experiments, please use the provided `Dockerfile` to build the Docker image. You can also pull the docker image by

    docker pull cheriell/cross-version-mpe:0.0.2

## Experiments

### Experiments in the ICLR 2024 tiny paper

Summary table of the experiments:

| Method  | Source Dataset | Target Dataset    | Main Script (in `experiments_iclr24tiny`)                |
| ------- | -------------- | ----------------- | ---------------------------------------------------------- |
| *Sup*   | --             | Schubert  Wagner  | {target_dataset_folder}/supervised.py                      |
| *T*     | MAESTRO        | Schubert, Wagner  | {target_dataset_folder}/teacher.py                         |
| *TS*    | MAESTRO        | Schubert, Wagner  | {target_dataset_folder}/teacher_student.py                 |
| *TSCV1* | MAESTRO        | Schubert, Wagner  | {target_dataset_folder}/teacher_student_cross_version_1.py |
| *TSCV2* | MAESTRO        | Schubert, Wagner  | {target_dataset_folder}/teacher_student_cross_version_2.py |

Before running the domain adaptation experiments, the teacher model is trained using script `experiments_iclr24tiny/teacher_maestro.py`.

Please refer to the `runme_iclr24tiny.sh` for detailed commands (including feature preparation) to run the experiments.

For the ICLR 2024 tiny paper, we uploaded the model checkpoints and pre-calculated features for the test sets at:
- Liu, L., & Weiß, C. (2024). Utilizing Cross-Version Consistency for Domain Adaptation: A Case Study on Music Audio (Pretrained Models and Features) (0.0.1). Zenodo. https://doi.org/10.5281/zenodo.10936492

### Experiments in the ICASSP 2025 paper

Summary table of the experiments:

| Method     | Target Dataset   | Main Config File (in `experiments_icassp25`) |
| ---------- | ---------------- | ---------------------------------------------- |
| *T*        | Wagner           | configs/ts/wagner_t.json                       |
| *TS*       | Wagner           | configs/ts/wagner_ts.json                      |
| *TSCV1*    | Wagner           | configs/ts/wagner_tscv.json                    |
| *TSCV2*    | Wagner           | configs/ts/wagner_tscv2.json                   |
| *AE*       | Schubert, Wagner | configs/aoe/{dataset}_aoe.json                 |
| *AE+TS*    | Schubert, Wagner | configs/aoe/{dataset}_aoets.json               |
| *AE+TSCV2* | Schubert, Wagner | configs/aoe/{dataset}_aoetscv.json             |

The source dataset is always MAESTRO. For the Wagner Ring Dataset, we used a different dataset split to the one used in the ICLR 2024 tiny paper, so we reran the experiments for T, TS, TSCV1 and TSCV2 on the Wagner Ring Dataset. The new dataset split is defined in `experiments_icassp25/dataset_splits/__init__.py`. You can also find the detailed dataset split for each dataset as Json files in the same folder.

Main python scripts are in folder `experiments_icassp25/main`.

Please refer to `runme_icassp25.sh` for detaild commands (including feature preparation) to run the experiments.