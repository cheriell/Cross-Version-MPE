# Cross-Version-MPE

Here we provide the accompanying code for our tiny paper 

- Lele Liu and Christof Weiss, "[Utilizing cross-version consistency for domain adaptation: A case study on music audio](https://openreview.net/forum?id=ZNg3YQQKWT),” in Tiny Papers Track at International Conference on Learning Representations (Tiny Papers @ ICLR), May 2024.

## Datasets and feature preparation

We use the [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro), the [Schubert Winterreise Dataset](https://zenodo.org/records/3968389) and the [Wagner Ring Dataset](https://zenodo.org/records/7672157) in our case study. 

During feature preparation, we calculate the HCQT spectrogram and binary pianoroll for each of the music performance, with a sampling rate of 22.05 kHz and a hop size of 512. We provide one example input-output pair of feature in the folder `example_precomputed_features`.

### Experiments

The teacher model is trained using script `experiments/teacher_maestro.py`.

Below are the scripts for each experiment mentioned in our paper:

| Method    | Source Dataset    | Target Dataset    | Script                                                           |
| --------- | ----------------- | ----------------- | ---------------------------------------------------------------- |
| *Sup*     | --                | Schubert  Wagner  | experiments/target_dataset/supervised.py                         |
| *T*       | MAESTRO           | Schubert, Wagner  | experiments/target_dataset/teacher.py                            |
| *TS*      | MAESTRO           | Schubert, Wagner  | experiments/target_dataset/teacher_student.py                    |
| *TSCV1*   | MAESTRO           | Schubert, Wagner  | experiments/target_dataset/teacher_student_cross_version_1.py    |
| *TSCV2*   | MAESTRO           | Schubert, Wagner  | experiments/target_dataset/teacher_student_cross_version_2.py    |

## Environments configuration

We use [synctoolbox](https://github.com/meinardmueller/synctoolbox) to calculate the alignment path between different versions. The python environment is the one provided by the toolbox (copied in file `environment-synctoolbox.yml`). Please use this python environment to run the feature preparation script `eature_preparation/prepare_cross_version_alignment.py`.

    conda env create -f environment-synctoolbox.yml
    conda activate synctoolbox

For running the experiments, please use the provided `Dockerfile` to build the Docker image.

## Running instruction

Please refer to the `runme.sh` for the whole reproduction pipeline.