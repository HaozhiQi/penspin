# --------------------------------------------------------
# Lessons from Learning to Spin “Pens”
# Written by Paper Authors
# Copyright (c) 2024 All Authors
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import os
import torch
import numpy as np
from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float, to_torch, quat_apply
from penspin.tasks.allegro_hand_hora import AllegroHandHora


class AllegroHandGrasp(AllegroHandHora):
    def __init__(self, config, sim_device, graphics_device_id, headless):
        super().__init__(config, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)
        self.saved_grasping_states = torch.zeros((0, 23), dtype=torch.float, device=self.device)
        # this dict is to define a canonical (mean) grasp when generating grasps
        self.canonical_pose_dict = {
            # for pencil (+z rotation), initialization is even harder
            # define 6 different canonical poses
            'pencil': [
                {'hand': [0.0344, 1.0537, 0.5184, 0.2920, 1.4615, 0.2352, 0.7243, -0.0436,
                          0.0974, 1.1978, 0.3126, 0.2979, 0.1340, 1.0228, 0.6625, 0.2327],
                 'object': [-0.01, -0.01, 0.62, 0.0000, -0.7071, -0.0000,  0.7071]},  # thumb | index + middle + ring
                {'hand': [-0.0578, 0.8993, 0.7449, 0.4767, 1.3089, 0.5699, 0.7037, 0.3857,
                          0.0299, 1.2735, 0.1415, 0.2507, 0.1457, 1.0319, 1.0376, 0.3283],
                 'object': [-0.01, -0.01, 0.62, 0.0000, -0.7071, -0.0000,  0.7071]},  # thumb | middle + ring
                {'hand': [0.0031, 1.0689, 0.8180, 0.1929, 1.2988, 0.7557, 0.5802, 0.3827,
                          0.0077, 1.2703, 0.1356, 0.3160, 0.1375, 1.3213, 0.5802, 0.2302],
                 'object': [-0.01, -0.01, 0.62, 0.2706, -0.6533, 0.2706, 0.6533]},  # thumb + index | middle + ring
                {'hand': [-0.1146, 1.1154, 0.4979, 0.1732, 0.8928, 1.1395, 0.5224, 1.1077,
                          -0.1250, 1.2618, 0.1522, 0.7024, 0.0226, 1.2307, 0.7128, 0.1524],
                 'object': [-0.01, -0.01, 0.62, 0.5, -0.5, 0.5, 0.5]},  # index | middle
                {'hand': [-0.0804, 1.3213, 0.3743, 0.0876, 1.2907, 0.7441, 0.4251, 0.3748,
                          -0.1412, 1.3882, 0.4888, 0.1689, -0.0802, 1.2987, 0.6577, 0.1455],
                 'object': [-0.01, -0.01, 0.62, 0.653281, -0.270598, 0.653281, 0.270598]},  # index | thumb + middle
                {'hand': [-0.1346, 1.1048, 0.5053, 0.1029, 1.2654, 0.6378, 0.8787, 0.0611,
                          -0.0573, 1.0228, 1.1360, 0.2587, 0.0253, 1.0228, 1.0845, 0.3828],
                 'object': [-0.01, -0.01, 0.62, 0.0000, -0.7071, -0.0000,  0.7071]},  # thumb | index
            ]}

        self.canonical_pose = self.canonical_pose_dict[self.canonical_pose_category]
        self.sampled_init_pose = torch.zeros((len(self.envs), 23), dtype=torch.float, device=self.device)

        # TODO: this is only for pencil prototype
        self.pencil_ends = [
            [0, 0, -0.76 / 2 * self.base_obj_scale],
            [0, 0, 0.76 / 2 * self.base_obj_scale]
        ]
        self.reset_steps = 0

    def reset_idx(self, env_ids):
        if self.randomize_pd_gains:
            self.p_gain[env_ids] = torch_rand_float(
                self.randomize_p_gain_lower, self.randomize_p_gain_upper, (len(env_ids), self.num_actions),
                device=self.device).squeeze(1)
            self.d_gain[env_ids] = torch_rand_float(
                self.randomize_d_gain_lower, self.randomize_d_gain_upper, (len(env_ids), self.num_actions),
                device=self.device).squeeze(1)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_allegro_hand_dofs * 2 + 5), device=self.device)

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0
        success = self.progress_buf[env_ids] == self.max_episode_length
        all_states = torch.cat([
            self.allegro_hand_dof_pos, self.root_state_tensor[self.object_indices, :7]
        ], dim=1)

        self.saved_grasping_states = torch.cat([self.saved_grasping_states, all_states[env_ids][success]])
        print('current cache size:', self.saved_grasping_states.shape[0])

        pose_threshold = int(eval(self.num_pose_per_cache[:-1]) * 1e3)
        if self.reset_steps % 200 == 0:
            cache_dir = '/'.join(['cache', self.grasp_cache_name, self.canonical_pose_category])
            os.makedirs(cache_dir, exist_ok=True)
            cache_name = f's{str(self.base_obj_scale).replace(".", "")}_{len(self.saved_grasping_states)}'
            cache_name = f'{cache_dir}/{cache_name}.npy'
            np.save(cache_name, self.saved_grasping_states.cpu().numpy())
        if len(self.saved_grasping_states) >= pose_threshold:
            cache_dir = '/'.join(['cache', self.grasp_cache_name, self.canonical_pose_category])
            os.makedirs(cache_dir, exist_ok=True)
            cache_name = f's{str(self.base_obj_scale).replace(".", "")}_{self.num_pose_per_cache}'
            cache_name = f'{cache_dir}/{cache_name}.npy'
            np.save(cache_name, self.saved_grasping_states[:pose_threshold].cpu().numpy())
            exit()

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_state[env_ids, 0:2]
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_state[env_ids, self.up_axis_idx]

        hand_randomize_amount = {
            'pencil': 0.05, 'thinpencil': 0.05,
        }[self.canonical_pose_category]
        obj_randomize_amount = {
            'pencil': [0.0, 0.0, 0.0], 'thinpencil': [0.01, 0.01, 0.0],
        }[self.canonical_pose_category]
        # Pencil Setting
        pose_ids = np.random.randint(0, len(self.canonical_pose), size=len(env_ids))
        hand_pose = to_torch([self.canonical_pose[pose_id]['hand'] for pose_id in pose_ids], device=self.device)
        hand_pose += hand_randomize_amount * rand_floats[:, 5:5 + self.num_allegro_hand_dofs]
        object_pose = to_torch([self.canonical_pose[pose_id]['object'] for pose_id in pose_ids], device=self.device)
        self.root_state_tensor[self.object_indices[env_ids], 0:7] = object_pose[:, 0:7]
        for i in range(3):
            self.root_state_tensor[self.object_indices[env_ids], i] += (obj_randomize_amount[i] * torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device))[:, 0]
        self.sampled_init_pose[env_ids] = torch.cat([hand_pose, self.root_state_tensor[self.object_indices[env_ids], :7]], dim=-1)

        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])
        object_indices = torch.unique(self.object_indices[env_ids]).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(object_indices), len(object_indices))

        self.allegro_hand_dof_pos[env_ids, :] = hand_pose
        self.allegro_hand_dof_vel[env_ids, :] = 0
        self.prev_targets[env_ids, :self.num_allegro_hand_dofs] = hand_pose
        self.cur_targets[env_ids, :self.num_allegro_hand_dofs] = hand_pose

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        if not self.torque_control:
            self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets), gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.progress_buf[env_ids] = 0
        self.obs_buf[env_ids] = 0
        self.rb_forces[env_ids] = 0
        self.priv_info_buf[env_ids, 0:3] = 0
        self.proprio_hist_buf[env_ids] = 0

        self.at_reset_buf[env_ids] = 1
        self.reset_steps += 1

    def compute_reward(self, actions):
        def list_intersect(li, hash_num):
            # 17 is the object index
            # 4, 8, 12, 16 are fingertip index
            # return number of contact with obj_id
            obj_id = self.rigid_body_states.shape[1] - 1
            query_list = [obj_id * hash_num + self.fingertip_handles[0], obj_id * hash_num + self.fingertip_handles[1], obj_id * hash_num + self.fingertip_handles[2], obj_id * hash_num + self.fingertip_handles[3]]
            return len(np.intersect1d(query_list, li))
        assert self.device == 'cpu'
        contacts = [self.gym.get_env_rigid_contacts(env) for env in self.envs]
        contact_list = [list_intersect(np.unique([c[2] * 10000 + c[3] for c in contact]), 10000) for contact in contacts]
        contact_condition = to_torch(contact_list, device=self.device)

        obj_pos = self.rigid_body_states[:, [-1], :3]
        finger_pos = self.rigid_body_states[:, self.fingertip_handles, :3]
        # the sampled pose need to satisfy (check 1 here):
        # 1) all fingertips is nearby objects
        if self.canonical_pose_category == 'pencil':
            cond1 = torch.ones(1)
        else:
            cond1 = (torch.sqrt(((obj_pos - finger_pos) ** 2).sum(-1)) < 0.1).all(-1)
        # 2) at least two fingers are in contact with object
        cond2 = contact_condition >= 2
        # 3) object does not fall after a few iterations
        cond3 = torch.greater(obj_pos[:, -1, -1], self.reset_z_threshold)
        cond = cond1.float() * cond2.float() * cond3.float()

        if self.canonical_pose_category == 'pencil':
            pencil_end_1 = self.object_pos + quat_apply(self.object_rot, to_torch(self.pencil_ends[0], device=self.device)[None].repeat(self.num_envs, 1))
            pencil_end_2 = self.object_pos + quat_apply(self.object_rot, to_torch(self.pencil_ends[1], device=self.device)[None].repeat(self.num_envs, 1))
            pencil_z_min = torch.min(pencil_end_1, pencil_end_2)[:, -1]
            pencil_z_max = torch.max(pencil_end_1, pencil_end_2)[:, -1]
            if self.init_pose_mode == "high":
                cond4 = torch.logical_and(pencil_z_min > 0.62, pencil_z_max < 0.65)
            elif self.init_pose_mode == "low":
                cond4 = torch.logical_and(pencil_z_min > 0.60, pencil_z_max < 0.63)
            else:
                raise NotImplementedError
            cond = cond * cond4.float()

        # reset if any of the above condition does not hold
        self.reset_buf[cond < 1] = 1
        self.reset_buf[self.progress_buf >= self.max_episode_length] = 1
