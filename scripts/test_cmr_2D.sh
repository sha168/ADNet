#!/bin/bash

# Specs.
SEED=2021
FOLD=0  # testing on this fold
DATA=<path_to_data>/CMR
PRETRAINED=results_cmr/train/fold${FOLD}/<model_name>
SAVE_FOLDER=results_cmr/test/fold${FOLD}
mkdir -p ${SAVE_FOLDER}

# Run.
python3 main_inference.py \
--data_root ${DATA} \
--save_root ${SAVE_FOLDER} \
--pretrained_root "${MODELPATH}" \
--dataset CMR \
--fold ${FOLD} \
--seed ${SEED}

# Note: EP2 is default, for EP1 set --EP1 True, --n_shot 3.

