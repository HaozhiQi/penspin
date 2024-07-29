#!/bin/bash
CACHE=$1

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

python train.py task=AllegroHandHora headless=False \
task.env.numEnvs=1 test=True \
task.env.object.type=cylinder_pencil-5-7 \
task.env.randomForceProbScalar=0.25 train.algo=PPO \
task.env.rotation_axis=+z \
task.env.genGraspCategory=pencil task.env.privInfo.enable_obj_orientation=True \
task.env.privInfo.enable_ft_pos=True task.env.privInfo.enable_obj_angvel=True \
task.env.randomization.randomizeScaleList=[0.28,0.29] task.env.grasp_cache_name=allegro_round_tip_thin \
task.env.asset.handAsset=assets/round_tip/allegro_hand_right_fsr_round_dense.urdf \
task.env.privInfo.enable_tactile=True train.ppo.priv_info=True task.env.hora.point_cloud_sampled_dim=100 \
task.env.numObservations=276 task.env.initPoseMode=low task.env.reset_height_threshold=0.605 \
task.env.reward.angvelClipMax=0.5 task.env.forceScale=2.0 \
task.env.reward.angvelPenaltyThres=1.0 \
task.env.enable_obj_ends=True \
wandb_activate=False \
checkpoint=outputs/AllegroHandHora/"${CACHE}"/stage1_nn/best*.pth \
${EXTRA_ARGS}
