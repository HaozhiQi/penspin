## Stage 1

We also provide scripts for baselines with different sensing capabilities and reward design.

* Without tactile information:
```
scripts/train_teacher.sh train.ppo.enable_tactile=False
```

* Without point cloud information:
```
scripts/train_teacher.sh task.env.hora.point_cloud_sampled_dim=100
```

* Without privileged information:
```
scripts/train_teacher.sh train.ppo.priv_info=False train.ppo.asymm_actor_critic=False
```

* Without z-reward:
```
scripts/train_teacher.sh task.env.reward.pencil_z_dist_penalty_scale=-1.0
```


## Stage 2

Here we also provide an implementation of [DAgger](https://arxiv.org/abs/1011.0686) as our baselines methods, which can be run by
* proprio input only:
```
scripts/train_student_sim.sh train.ppo.is_demon=False train.demon_path=ORACLE_CHECKPOINT_PATH train.ppo.input_mode=proprio
```

* proprio and tactile input:
```
scripts/train_student_sim.sh train.ppo.is_demon=False train.demon_path=ORACLE_CHECKPOINT_PATH train.ppo.input_mode=proprio-tactile
```

* proprio and obj-ends input:
```
scripts/train_student_sim.sh train.ppo.is_demon=False train.demon_path=ORACLE_CHECKPOINT_PATH train.ppo.input_mode=proprio-ends
```

* proprio, tactile and obj-ends input:
```
scripts/train_student_sim.sh train.ppo.is_demon=False train.demon_path=ORACLE_CHECKPOINT_PATH train.ppo.input_mode=proprio-tactile-ends
```
