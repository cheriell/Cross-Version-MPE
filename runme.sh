# After calculating the HCQT and pianoroll features, run this script to train the teacher and student models.

##############################################################
# Supervised (Sup)
##############################################################
python experiments/Schubert_Winterreise/supervised.py
python experiments/Wagner_Ring_allperf/supervised.py

##############################################################
# Teacher model (T)
##############################################################
python experiments/teacher_maestro.py
python experiments/Schubert_Winterreise/teacher.py
python experiments/Wagner_Ring_allperf/teacher.py

##############################################################
# Teacher-Student model (TS)
##############################################################
# Prepare teacher annotations (change the dataset name and teacher model path in the script before running it)
python feature_preparation/prepare_teacher_annotations.py
# Train the student models
python experiments/Schubert_Winterreise/teacher_student.py
python experiments/Wagner_Ring_allperf/teacher_student.py

##############################################################
# Teacher-Student model with cross-version training (TS-CV1 and TS-CV2)
##############################################################
# Prepare cross-version alignment
# Change the dataset name in the script before running the following line
python feature_preparation/prepare_cross_version_alignment.py
# Train the student models using the cross-version pairs
python experiments/Schubert_Winterreise/teacher_student_cross_version_1.py
python experiments/Wagner_Ring_allperf/teacher_student_cross_version_1.py
python experiments/Schubert_Winterreise/teacher_student_cross_version_2.py
python experiments/Wagner_Ring_allperf/teacher_student_cross_version_2.py