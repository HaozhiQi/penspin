#!/bin/bash
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:0:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=0 \
python train.py task=AllegroHandHora headless=True \
seed=0 task.env.randomForceProbScalar=0.25 train.algo=DemonTrain \
train.ppo.proprio_adapt=False train.ppo.output_name=AllegroHandHora/pencilup/240603_student_pretrain \
experiment=pencilup-demo-thin-student-pretrain task.env.rotation_axis=+z \
task.env.reward.pencil_z_dist_penalty_scale=-1.0 task.env.forceScale=2 \
task.env.randomization.obs_noise_t_scale=0.01 \
task.env.randomization.obs_noise_e_scale=0.02 \
task.env.object.type=cylinder_pencil-5-7 task.env.genGraspCategory=pencil \
task.env.privInfo.enable_obj_orientation=True task.env.privInfo.enable_ft_pos=True \
task.env.privInfo.enable_obj_angvel=True train.ppo.max_agent_steps=10000000000 \
task.env.randomization.randomizeScaleList=[0.29] task.env.grasp_cache_name=allegro_round_tip_thin \
task.env.privInfo.enable_tactile=True task.env.hora.point_cloud_sampled_dim=100 \
task.env.initPoseMode=low task.env.reset_height_threshold=0.605 task.env.reward.angvelClipMax=0.5 \
train.ppo.priv_info=False \
train.ppo.horizon_length=512 task.env.numEnvs=48 train.ppo.minibatch_size=4096 \
train.ppo.distill=True train.ppo.use_l1=True train.ppo.enable_latent_loss=False \
task.env.enable_obj_ends=True \
train.ppo.proprio_mode=True train.ppo.proprio_len=30 train.ppo.learning_rate=1e-3 train.ppo.input_mode=proprio \
train.ppo.is_demon=True wandb_activate=False \
train.demon_path=last.pth \
${EXTRA_ARGS}