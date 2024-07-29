# Lessons from Learning to Spin “Pens”

<p align="center">
  <img src="assets/teaser.gif" width="1000"/>
</p>

This repository contains a reference PyTorch implementation of the paper:

<b>Lessons from Learning to Spin “Pens”</b> <br>
[Jun Wang*](https://wang59695487.github.io/),
[Ying Yuan*](https://yingyuan0414.github.io/),
[Haichuan Che*](https://www.linkedin.com/in/haichuan-che-7338721b1/),
[Haozhi Qi*](https://haozhi.io/),
[Yi Ma](http://people.eecs.berkeley.edu/~yima/),
[Jitendra Malik](https://people.eecs.berkeley.edu/~malik/),
[Xiaolong Wang](https://xiaolonw.github.io/) <br>
[[Website](https://penspin.github.io/)]

## Installation

See [installation instructions](docs/install.md).

## Introduction

Our pen spinning method contains the following four steps.
1. Learn a oracle policy with privileged information, point-clouds, and tactile sensor output with RL in simulation.
2. Learn a student policy using the rollout of the oracle policy, also in simulation.
3. Rollout trajectories generated by the oracle policy in a real robot, with initial state distribution matched. The success trajectories are collected while failures are discarded.
4. Finetune the student policy in step 2 with the real-world successful trajectories.

The following session only provides example script of our method. For baselines, checkout [baselines](docs/baseline.md).

## Step 0: Visualize a Pre-trained Oracle Policy

```
cd outputs/AllegroHandHora
gdown 1LCRFE6lvKSUDPpUfEATOmpDUPDbB7n8d
unzip demo.zip -d ./
cd ../../
scripts/vis_teacher.sh demo
```


## Step 1: Oracle Policy training

To train an oracle policy $f$ with RL, run

```
# 0 is GPU is
# 42 is experiment seed
scripts/train_teacher.sh 0 42 output_name
```

After training your oracle policy, you can visualize it as follows:
```
scripts/vis_teacher.sh output_name
```

## Step 2: Student Policy Pretraining

In this section, we train a proprioceptive student policy by distilling from our trained oracle policy $f$.

Note we use the teacher rollout to train student policy, in contrast to DAgger in previous works.

```
scripts/train_student_sim.sh train.ppo.is_demon=True train.demon_path=ORACLE_CHECKPOINT_PATH 
```
We have provided a reference teacher checkpoint in [Google Drive](https://drive.google.com/file/d/1LCRFE6lvKSUDPpUfEATOmpDUPDbB7n8d/view?usp=sharing).

## Step 3: Open-Loop Replay in Real Hardware

To generate open-loop replay data for the student policy $\pi$, run
```
python real/robot_controller/teacher_replay.py --data-collect --exp=0 --replay_data_dir=REPLAY_DATA_DIR
```
where `REPLAY_DATA_DIR` is the directory to save the replay data.

Then process the replay data.

## Step 4: Real-world Fine-tuning

To fine-tune the student policy $\pi$ using real data, run
```
scripts/finetune_ppo.sh --real-dataset-folder=REAL_DATA_PATH --checkpoint-path=YOUR_CHECKPOINTPATH
```

## Real Data Download
Please download the real reference data from [Google Drive](https://drive.google.com/drive/folders/1TAMAvqLp3b5vEmdyrdcgW0kBW1GAxoyy?usp=sharing).
```
Real data:
  real_data.h5 is in the format of h5 file, which contains the following keys:
  -replay_demon_{idx}: the idx-th replay demonstration data
    - qpos: the current qpos of the robot
    - action: the delta action applied to the robot
    - current_target_qpos: the target qpos of the robot

  real_data_full.h5 is a full version of real_data.h5, which contains the following keys:
  -replay_demon_{idx}: the idx-th replay demonstration data
    - qpos: the current qpos of the robot
    - action: the delta action applied to the robot
    - current_target_qpos: the target qpos of the robot
    - rgb_ori: the original rgb image
    - rgb_c2d: the rgb image after camera2depth image processing
    - depth: the depth image
    - pc: the point cloud
    - obj_ends: the position of object ends 
```

## Acknowledgement

Note: This repository is built based on [Hora](https://github.com/HaozhiQi/hora) and [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs).

## Citing

If you find **PenSpin** or this codebase helpful in your research, please consider citing:

```
@article{wang2024penspin,
  author={Wang, Jun and Yuan, Ying and Che, Haichuan and Qi, Haozhi and Ma, Yi and Malik, Jitendra and Wang, Xiaolong},
  title={Lessons from Learning to Spin “Pens”},
  journal={},
  year={2024}
}
```