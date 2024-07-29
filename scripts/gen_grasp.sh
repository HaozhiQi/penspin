#!/bin/bash

GPUS=$1
SCALE=$2

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python gen_grasp.py task=AllegroHandGrasp headless=True pipeline=cpu \
task.env.numEnvs=20000 test=True \
task.env.controller.controlFrequencyInv=8 task.env.episodeLength=40 \
task.env.controller.torque_control=False task.env.genGrasps=True task.env.baseObjScale="${SCALE}" \
task.env.genGraspCategory=pencil \
task.env.object.type=cylinder_pencil-5-7 \
task.env.randomization.randomizeMass=True task.env.randomization.randomizeMassLower=0.01 task.env.randomization.randomizeMassUpper=0.02 \
task.env.randomization.randomizeCOM=False \
task.env.randomization.randomizeFriction=False \
task.env.randomization.randomizePDGains=False \
task.env.randomization.randomizeScale=False \
train.ppo.priv_info=True task.env.grasp_cache_name=allegro_round_tip_thin \
task.env.reset_height_threshold=0.6 \
${EXTRA_ARGS}