# Cross-Version-MPE

Here we provide the accompanying code for our tiny paper 

- Lele Liu and Christof Weiß, "[Utilizing cross-version consistency for domain adaptation: A case study on music audio](https://openreview.net/forum?id=ZNg3YQQKWT),” in Tiny Papers Track at International Conference on Learning Representations (Tiny Papers @ ICLR), 2024.
- Lele Liu and Christof Weiß, "Unsupervised Domain Adaptation for Music Transcription: Exploiting Cross-Version Consistency," accepted to the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2025. (Code will be merged soon.)

## Datasets and feature preparation

We use the [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro), the [Schubert Winterreise Dataset](https://zenodo.org/records/3968389) and the [Wagner Ring Dataset](https://zenodo.org/records/7672157) in our case study. 

During feature preparation, we calculate the HCQT spectrogram and binary pianoroll for each of the music performance, with a sampling rate of 22.05 kHz and a hop size of 512.
The feature preparation code can be found [HERE](https://github.com/christofw/multipitch_architectures/blob/master/01_precompute_features.ipynb).
We provide one example input-output pair of feature in the folder `example_precomputed_features`.

### Experiments

The teacher model is trained using script `experiments/teacher_maestro.py`.

Below are the scripts for each experiment mentioned in our paper:

| Method    | Source Dataset    | Target Dataset    | Script                                                           |
| --------- | ----------------- | ----------------- | ---------------------------------------------------------------- |
| *Sup*     | --                | Schubert  Wagner  | experiments/<target_dataset>/supervised.py                         |
| *T*       | MAESTRO           | Schubert, Wagner  | experiments/<target_dataset>/teacher.py                            |
| *TS*      | MAESTRO           | Schubert, Wagner  | experiments/<target_dataset>/teacher_student.py                    |
| *TSCV*    | MAESTRO           | Schubert, Wagner  | experiments/<target_dataset>/teacher_student_cross_version_1.py    |
| *TSCV2*   | MAESTRO           | Schubert, Wagner  | experiments/<target_dataset>/teacher_student_cross_version_2.py    |

## Environments configuration

We use [synctoolbox](https://github.com/meinardmueller/synctoolbox) to calculate the alignment path between different versions. The python environment is the one provided by the toolbox (copied in file `environment-synctoolbox.yml`). Please use this python environment to run the feature preparation script `eature_preparation/prepare_cross_version_alignment.py`.

For running the experiments, please use the provided `Dockerfile` to build the Docker image. You can also pull the docker image by

    docker pull cheriell/cross-version-mpe:0.0.2

## Running instruction

Please refer to the `runme.sh` for the whole reproduction pipeline.

For reproducibility, we uploaded the model checkpoints and pre-calculated features for the test sets at:
- Liu, L., & Weiß, C. (2024). Utilizing Cross-Version Consistency for Domain Adaptation: A Case Study on Music Audio (Pretrained Models and Features) (0.0.1). Zenodo. https://doi.org/10.5281/zenodo.10936492


