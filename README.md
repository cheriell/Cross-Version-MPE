# Cross-Version-MPE

Accompanying code for paper submission "Utilizing Cross-Version Consistency for Domain Adaptation: A Case Study on Music Audio".

## Datasets and feature preparation

We use the [Schubert Winterreise Dataset](https://zenodo.org/records/3968389) and the [Wagner Ring Dataset](https://zenodo.org/records/7672157) in our case study. 

During feature preparation, we calculate the HCQT spectrogram and binary pianoroll for each of the music performance. We provide one example input-output pair of feature in the folder `example_precomputed_features`.

### Experiments

The teacher model is trained using script `experiments/teacher_maestro.py`.

Below are the scripts for each experiment mentioned in our paper:

| Method    | Source Dataset    | Target Dataset    | Script                                                           |
| --------- | ----------------- | ----------------- | ---------------------------------------------------------------- |
| *Sup*     | --                | Schubert  Wagner  | experiments/target_dataset/supervised.py                         |
| *T*       | MAESTRO           | Schubert, Wagner  | experiments/target_dataset/teacher.py                            |
| *TS*      | MAESTRO           | Schubert, Wagner  | experiments/target_dataset/teacher_student.py                    |
| *TS-CV1*  | MAESTRO           | Schubert, Wagner  | experiments/target_dataset/teacher_student_cross_version_1.py    |
| *TS-CV2*  | MAESTRO           | Schubert, Wagner  | experiments/target_dataset/teacher_student_cross_version_2.py    |

## Environments configuration

We use [synctoolbox](https://github.com/meinardmueller/synctoolbox) to calculate the alignment path between different versions. The python environment is the one provided by the toolbox (copied in file `environment-synctoolbox.yml`). Please use this python environment to run the feature preparation script `feature_preparation/prepare_cross_version_alignment.py`

    conda env create -f environment-synctoolbox.yml
    conda activate synctoolbox

For reproduction purposes, we provide the python environment for running our experiments (`environment-exp.yml`).

    conda env create -f environment-exp.yml
    conda activate exp

## Running instruction

Please refer to the `runme.sh` for the whole reproduction pipeline.