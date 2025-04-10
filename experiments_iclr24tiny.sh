# After calculating the HCQT and pianoroll features, you can follow the instructions below to run the experiments.

##############################################################
# Supervised (Sup)
##############################################################

# Trian and evaluate the supervised model on the Schubert Winterreise Dataset
python experiments_iclr24tiny/Schubert_Winterreise/supervised.py

# Train and evaluate the supervised model on the Wagner Ring Dataset
python experiments_iclr24tiny/Wagner_Ring_allperf/supervised.py


##############################################################
# Teacher model (T)
##############################################################

# Train the teacher model on the MAESTRO dataset
python experiments_iclr24tiny/teacher_maestro.py

# Evaluate the teacher model on the Schubert Winterreise Dataset
python experiments_iclr24tiny/Schubert_Winterreise/teacher.py

# Evaluate the teacher model on the Wagner Ring Dataset
python experiments_iclr24tiny/Wagner_Ring_allperf/teacher.py


##############################################################
# Teacher-Student model (TS)
##############################################################

# Prepare teacher annotations (change the dataset name and teacher model path in the script before running it)
python experiments_iclr24tiny/feature_preparation/prepare_teacher_annotations.py

# Train and evaluate the student models
python experiments_iclr24tiny/Schubert_Winterreise/teacher_student.py
python experiments_iclr24tiny/Wagner_Ring_allperf/teacher_student.py

##############################################################
# Teacher-Student model with cross-version training (TSCV1 and TSCV2)
##############################################################

# Prepare cross-version alignment
# Change the dataset name in the script before running the following command
python experiments_iclr24tiny/feature_preparation/prepare_cross_version_alignment.py

## Train and evaluate the student models using the cross-version pairs

# TSCV1
python experiments_iclr24tiny/Schubert_Winterreise/teacher_student_cross_version_1.py
python experiments_iclr24tiny/Wagner_Ring_allperf/teacher_student_cross_version_1.py

# TSCV2
python experiments_iclr24tiny/Schubert_Winterreise/teacher_student_cross_version_2.py
python experiments_iclr24tiny/Wagner_Ring_allperf/teacher_student_cross_version_2.py