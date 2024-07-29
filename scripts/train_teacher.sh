#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=AllegroHandHora headless=True seed=${SEED} \
task.env.object.type=cylinder_pencil-5-7 \
experiment=rl \
task.env.randomForceProbScalar=0.25 train.algo=PPO train.ppo.proprio_adapt=False \
task.env.rotation_axis=+z task.env.reward.pencil_z_dist_penalty_scale=-1.0 \
task.env.genGraspCategory=pencil task.env.privInfo.enable_obj_orientation=True \
task.env.privInfo.enable_ft_pos=True task.env.privInfo.enable_obj_angvel=True \
train.ppo.horizon_length=12 train.ppo.max_agent_steps=10000000000 \
task.env.randomization.randomizeScaleList=[0.28,0.29] task.env.grasp_cache_name=allegro_round_tip_thin \
task.env.privInfo.enable_tactile=True train.ppo.priv_info=True task.env.hora.point_cloud_sampled_dim=100 \
task.env.numObservations=276 task.env.initPoseMode=low task.env.reset_height_threshold=0.605 \
task.env.reward.angvelClipMax=0.5 task.env.forceScale=2.0 \
task.env.enable_obj_ends=True wandb_activate=False \
train.ppo.output_name=AllegroHandHora/${CACHE} \
${EXTRA_ARGS}
