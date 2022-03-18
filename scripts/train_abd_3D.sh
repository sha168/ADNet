#!/bin/bash

# Specs.
SEED=2021
FOLD=0  # indicate testing fold (will be trained on the rest!)
RUNS=1  # number of runs (repetitions)
DATA=<path_to_data>/CHAOST2
SAVE_FOLDER=results_abd_3d/train/fold${FOLD}

# Run.
mkdir -p ${SAVE_FOLDER}
for _ in $(seq 1 ${RUNS})
do
  python3 main_train_3D.py \
  --data_root ${DATA} \
  --save_root ${SAVE_FOLDER} \
  --dataset CHAOST2 \
  --steps 50000 \
  --n_sv 5000 \
  --fold ${FOLD} \
  --seed ${SEED}
done

