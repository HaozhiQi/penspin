# --------------------------------------------------------
# Lessons from Learning to Spin “Pens”
# Written by Paper Authors
# Copyright (c) 2024 All Authors
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
from typing import Optional
import torch
import omegaconf
import numpy as np

from glob import glob
from collections import OrderedDict

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import quat_conjugate, quat_mul, to_torch, quat_apply, tensor_clamp, torch_rand_float, quat_from_euler_xyz

from penspin.utils.point_cloud_prep import sample_cylinder, sample_cuboid
from .base.vec_task import VecTask
from penspin.utils.misc import tprint


class AllegroHandHora(VecTask):
    def __init__(self, config, sim_device, graphics_device_id, headless):
        self.config = config
        # before calling init in VecTask, need to do
        # 1. setup randomization
        self._setup_domain_rand_config(config['env']['randomization'])
        # 2. setup privileged information
        self._setup_priv_option_config(config['env']['privInfo'])
        # 3. setup object assets
        self._setup_object_info(config['env']['object'])
        # 4. setup rewards
        self._setup_reward_config(config['env']['reward'])
        # unclassified config
        self.obs_with_binary_contact = config['env']['obs_with_binary_contact']
        self.base_obj_scale = config['env']['baseObjScale']
        self.save_init_pose = config['env']['genGrasps']
        self.aggregate_mode = self.config['env']['aggregateMode']
        self.up_axis = 'z'
        self.rotation_axis = config['env']['rotation_axis']
        self.reset_z_threshold = self.config['env']['reset_height_threshold']
        self.grasp_cache_name = self.config['env']['grasp_cache_name']
        self.canonical_pose_category = config['env']['genGraspCategory']
        self.num_pose_per_cache = '50k'
        self.with_camera = config['env']['enableCameraSensors']
        self.enable_obj_ends = config['env']['enable_obj_ends']
        self.init_pose_mode = config['env']['initPoseMode']

        # Important: map CUDA device IDs to Vulkan ones.
        graphics_device_id = 0

        super().__init__(config, sim_device, graphics_device_id, headless)

        self.eval_done_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.debug_viz = self.config['env']['enableDebugVis']
        self.max_episode_length = self.config['env']['episodeLength']
        self.dt = self.sim_params.dt

        if self.viewer:
            cam_pos = gymapi.Vec3(0.0, 0.4, 1.5)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        force_sensor = self.gym.acquire_force_sensor_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.allegro_hand_default_dof_pos = torch.zeros(self.num_allegro_hand_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        print("Contact Tensor Dimension", self.contact_forces.shape)
        self.allegro_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_allegro_hand_dofs]
        self.allegro_hand_dof_pos = self.allegro_hand_dof_state[..., 0]
        self.allegro_hand_dof_vel = self.allegro_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self._refresh_gym()

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        # object apply random forces parameters
        self.force_scale = self.config['env'].get('forceScale', 0.0)
        self.random_force_prob_scalar = self.config['env'].get('randomForceProbScalar', 0.0)
        self.force_decay = self.config['env'].get('forceDecay', 0.99)
        self.force_decay_interval = self.config['env'].get('forceDecayInterval', 0.08)
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

        self.last_contacts = torch.zeros((self.num_envs, self.num_contacts), dtype=torch.float, device=self.device)
        self.contact_thresh = torch.zeros((self.num_envs, self.num_contacts), dtype=torch.float, device=self.device)

        if self.randomize_scale and self.scale_list_init:
            self.saved_grasping_states = {}
            for s in self.randomize_scale_list:
                cache_name = '_'.join([self.grasp_cache_name, 'grasp', self.canonical_pose_category,
                                       self.num_pose_per_cache, f's{str(s).replace(".", "")}'])
                cache_name_tmp = '/'.join([self.grasp_cache_name, self.canonical_pose_category,
                                           f's{str(s).replace(".", "")}_{self.num_pose_per_cache}'])
                print(cache_name_tmp)
                if os.path.exists(f'cache/{cache_name_tmp}.npy'):
                    self.saved_grasping_states[str(s)] = torch.from_numpy(np.load(f'cache/{cache_name_tmp}.npy')).float().to(self.device)
                    print(cache_name_tmp)
                else:
                    self.saved_grasping_states[str(s)] = torch.from_numpy(np.load(f'cache/{cache_name}.npy')).float().to(self.device)
                    print(cache_name)
        else:
            assert self.save_init_pose

        self.rot_axis_buf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.rot_axis_task = None
        sign, axis = self.rotation_axis[0], self.rotation_axis[1]
        axis_index = ['x', 'y', 'z'].index(axis)
        self.rot_axis_buf[:, axis_index] = 1
        self.rot_axis_buf[:, axis_index] = -self.rot_axis_buf[:, axis_index] if sign == '-' else self.rot_axis_buf[:, axis_index]

        # useful buffers
        self.init_pose_buf = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        # there is an extra dim [self.control_freq_inv] because we want to get a mean over multiple control steps
        self.torques = torch.zeros((self.num_envs, self.control_freq_inv, self.num_actions), device=self.device, dtype=torch.float)
        self.dof_vel_finite_diff = torch.zeros((self.num_envs, self.control_freq_inv, self.num_dofs), device=self.device, dtype=torch.float)

        # --- calculate velocity at control frequency instead of simulated frequency
        self.object_pos_prev = self.object_pos.clone()
        self.object_rot_prev = self.object_rot.clone()
        self.ft_pos_prev = self.fingertip_pos.clone()
        self.ft_rot_prev = self.fingertip_orientation.clone()
        self.dof_vel_prev = self.dof_vel_finite_diff.clone()

        self.obj_linvel_at_cf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.obj_angvel_at_cf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.ft_linvel_at_cf = torch.zeros((self.num_envs, 4 * 3), device=self.device, dtype=torch.float)
        self.ft_angvel_at_cf = torch.zeros((self.num_envs, 4 * 3), device=self.device, dtype=torch.float)
        self.dof_acc = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        # ----

        assert type(self.p_gain) in [int, float] and type(self.d_gain) in [int, float], 'assume p_gain and d_gain are only scalars'
        self.p_gain = torch.ones((self.num_envs, self.num_actions), device=self.device, dtype=torch.float) * self.p_gain
        self.d_gain = torch.ones((self.num_envs, self.num_actions), device=self.device, dtype=torch.float) * self.d_gain

        # debug and understanding statistics
        self.evaluate = self.config['on_evaluation']
        self.evaluate_cache_name = self.config['eval_cache_name']
        self.stat_sum_rewards = [0 for _ in self.object_type_list]  # all episode reward
        self.stat_sum_episode_length = [0 for _ in self.object_type_list]  # average episode length
        self.stat_sum_rotate_rewards = [0 for _ in self.object_type_list]  # rotate reward, with clipping
        self.stat_sum_rotate_penalty = [0 for _ in self.object_type_list]  # rotate penalty with clipping
        self.stat_sum_unclip_rotate_rewards = [0 for _ in self.object_type_list]  # rotate reward, with clipping
        self.stat_sum_unclip_rotate_penalty = [0 for _ in self.object_type_list]  # rotate penalty with clipping
        self.extrin_log = []
        self.env_evaluated = [0 for _ in self.object_type_list]
        self.evaluate_iter = 0

        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._create_ground_plane()
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self._create_object_asset()
        allegro_hand_dof_props = self._parse_hand_dof_props()
        hand_pose, obj_pose = self._init_object_pose()

        # compute aggregate size
        self.num_allegro_hand_bodies = self.gym.get_asset_rigid_body_count(self.hand_asset)
        self.num_allegro_hand_shapes = self.gym.get_asset_rigid_shape_count(self.hand_asset)
        max_agg_bodies = self.num_allegro_hand_bodies + 2
        max_agg_shapes = self.num_allegro_hand_shapes + 2

        self.envs = []
        self.vid_record_tensor = None  # Used for record video during training, NOT FOR POLICY OBSERVATION
        self.object_init_state = []

        self.hand_indices = []
        self.hand_actors = []
        self.object_indices = []
        self.object_type_at_env = []

        self.obj_point_clouds = []

        allegro_hand_rb_count = self.gym.get_asset_rigid_body_count(self.hand_asset)
        object_rb_count = 1
        self.object_rb_handles = list(range(allegro_hand_rb_count, allegro_hand_rb_count + object_rb_count))

        for i in range(num_envs):
            tprint(f'{i} / {num_envs}')
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies * 20, max_agg_shapes * 20, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            hand_actor = self.gym.create_actor(env_ptr, self.hand_asset, hand_pose, 'hand', i, -1, 1)
            self.gym.set_actor_dof_properties(env_ptr, hand_actor, allegro_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)
            self.hand_actors.append(hand_actor)

            # add object
            eval_object_type = self.config['env']['object']['evalObjectType']
            if eval_object_type is None:
                object_type_id = np.random.choice(len(self.object_type_list), p=self.object_type_prob)
            else:
                object_type_id = self.object_type_list.index(eval_object_type)

            self.object_type_at_env.append(object_type_id)
            object_asset = self.object_asset_list[object_type_id]

            object_handle = self.gym.create_actor(env_ptr, object_asset, obj_pose, 'object', i, 0, 2)
            self.object_init_state.append([
                obj_pose.p.x, obj_pose.p.y, obj_pose.p.z,
                obj_pose.r.x, obj_pose.r.y, obj_pose.r.z, obj_pose.r.w,
                0, 0, 0, 0, 0, 0
            ])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            self.obj_scale = self.base_obj_scale
            if self.randomize_scale:
                num_scales = len(self.randomize_scale_list)
                self.obj_scale = np.random.uniform(
                    self.randomize_scale_list[i % num_scales] - 0.025,
                    self.randomize_scale_list[i % num_scales] + 0.025
                )
            self.gym.set_actor_scale(env_ptr, object_handle, self.obj_scale)
            self._update_priv_buf(env_id=i, name='obj_scale', value=self.obj_scale)

            obj_com = [0, 0, 0]
            if self.randomize_com:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                assert len(prop) == 1
                obj_com = [np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                           np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                           np.random.uniform(self.randomize_com_lower, self.randomize_com_upper)]
                prop[0].com.x, prop[0].com.y, prop[0].com.z = obj_com
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
            self._update_priv_buf(env_id=i, name='obj_com', value=obj_com)

            obj_friction = 1.0
            obj_restitution = 0.0  # default is 0
            # TODO: bad engineering because of urgent modification
            if self.randomize_friction:
                rand_friction = np.random.uniform(self.randomize_friction_lower, self.randomize_friction_upper)
                obj_restitution = np.random.uniform(0, 1)

                hand_props = self.gym.get_actor_rigid_shape_properties(env_ptr, hand_actor)
                for p in hand_props:
                    p.friction = rand_friction
                    p.restitution = obj_restitution
                self.gym.set_actor_rigid_shape_properties(env_ptr, hand_actor, hand_props)

                object_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
                for p in object_props:
                    p.friction = rand_friction
                    p.restitution = obj_restitution
                self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_props)
                obj_friction = rand_friction
            self._update_priv_buf(env_id=i, name='obj_friction', value=obj_friction)
            self._update_priv_buf(env_id=i, name='obj_restitution', value=obj_restitution)

            if self.randomize_mass:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                for p in prop:
                    p.mass = np.random.uniform(self.randomize_mass_lower, self.randomize_mass_upper)
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
                self._update_priv_buf(env_id=i, name='obj_mass', value=prop[0].mass)
            else:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                self._update_priv_buf(env_id=i, name='obj_mass', value=prop[0].mass)

            if self.point_cloud_sampled_dim > 0:
                self.obj_point_clouds.append(self.asset_point_clouds[object_type_id] * self.obj_scale)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # for training record, visualized in tensorboard
            if self.with_camera:
                self.vid_record_tensor = self._create_camera(env_ptr)

            self.envs.append(env_ptr)

        sensor_handles = [self.gym.find_actor_rigid_body_handle(
            env_ptr, hand_actor, sensor_name
        ) for sensor_name in self.contact_sensor_names]
        self.sensor_handle_indices = to_torch(sensor_handles, dtype=torch.int64)
        self.obj_point_clouds = to_torch(np.array(self.obj_point_clouds), device=self.device, dtype=torch.float)
        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.object_type_at_env = to_torch(self.object_type_at_env, dtype=torch.long, device=self.device)

    def _create_camera(self, env_ptr) -> torch.Tensor:
        """Create a camera in a particular environment. Should be called in _create_envs."""
        camera_props = gymapi.CameraProperties()
        camera_props.width = 256
        camera_props.height = 256
        camera_props.enable_tensors = True
        
        camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)

        cam_pos = gymapi.Vec3(0.0, 0.2, 0.75)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.5)

        self.gym.set_camera_location(camera_handle, env_ptr, cam_pos, cam_target)
        # obtain camera tensor
        vid_record_tensor = self.gym.get_camera_image_gpu_tensor(
            self.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR
        )
        # wrap camera tensor in a pytorch tensor
        vid_record_tensor.device = 0
        torch_vid_record_tensor = gymtorch.wrap_tensor(vid_record_tensor)
        assert torch_vid_record_tensor.shape == (camera_props.height, camera_props.width, 4)

        return torch_vid_record_tensor

    def reset_idx(self, env_ids):
        if self.randomize_pd_gains:
            self.p_gain[env_ids] = torch_rand_float(
                self.randomize_p_gain_lower, self.randomize_p_gain_upper, (len(env_ids), self.num_actions),
                device=self.device).squeeze(1)
            self.d_gain[env_ids] = torch_rand_float(
                self.randomize_d_gain_lower, self.randomize_d_gain_upper, (len(env_ids), self.num_actions),
                device=self.device).squeeze(1)

        self.random_obs_noise_e[env_ids] = torch.normal(0, self.random_obs_noise_e_scale, size=(len(env_ids), self.num_dofs), device=self.device, dtype=torch.float)
        self.random_action_noise_e[env_ids] = torch.normal(0, self.random_action_noise_e_scale, size=(len(env_ids), self.num_dofs), device=self.device, dtype=torch.float)
        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        num_scales = len(self.randomize_scale_list)
        for n_s in range(num_scales):
            s_ids = env_ids[(env_ids % num_scales == n_s).nonzero(as_tuple=False).squeeze(-1)]
            if len(s_ids) == 0:
                continue
            obj_scale = self.randomize_scale_list[n_s]
            scale_key = str(obj_scale)
            # single object (category) case:
            sampled_pose_idx = np.random.randint(self.saved_grasping_states[scale_key].shape[0], size=len(s_ids))
            sampled_pose = self.saved_grasping_states[scale_key][sampled_pose_idx].clone()
            object_pose_noise = torch.normal(0, self.random_pose_noise, size=(sampled_pose.shape[0], 7), device=self.device, dtype=torch.float)
            object_pose_noise[:, 3:] = 0  # disable rotation noise
            self.root_state_tensor[self.object_indices[s_ids], :7] = sampled_pose[:, 16:] + object_pose_noise
            self.root_state_tensor[self.object_indices[s_ids], 7:13] = 0
            pos = sampled_pose[:, :16]
            self.allegro_hand_dof_pos[s_ids, :] = pos
            self.allegro_hand_dof_vel[s_ids, :] = 0
            self.prev_targets[s_ids, :self.num_allegro_hand_dofs] = pos
            self.cur_targets[s_ids, :self.num_allegro_hand_dofs] = pos
            self.init_pose_buf[s_ids, :] = pos

        object_indices = torch.unique(self.object_indices[env_ids]).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(object_indices), len(object_indices))
        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        if not self.torque_control:
            self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets), gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        # reset tactile
        self.contact_thresh[env_ids] = 0.05
        self.last_contacts[env_ids] = 0.0

        self.progress_buf[env_ids] = 0
        self.obs_buf[env_ids] = 0
        self.rb_forces[env_ids] = 0
        self.priv_info_buf[env_ids, 0:3] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.tactile_hist_buf[env_ids] = 0
        self.noisy_quaternion_buf[env_ids] = 0
        self.dof_vel_finite_diff[:] = 0
        self.at_reset_buf[env_ids] = 1

    def compute_observations(self):
        self._refresh_gym()
        # observation noise
        random_obs_noise_t = torch.normal(0, self.random_obs_noise_t_scale, size=self.allegro_hand_dof_pos.shape, device=self.device, dtype=torch.float)
        noisy_joint_pos = random_obs_noise_t + self.random_obs_noise_e + self.allegro_hand_dof_pos
        
        # update tactile sensing
        if self.config['env']['privInfo']['enable_tactile']:
            contacts = self.contact_forces.clone()
            contacts = contacts[:, self.sensor_handle_indices, :]
            contacts = torch.norm(contacts, dim=-1)
            contacts = torch.where(contacts >= self.contact_thresh, 1.0, 0.0)
            latency_samples = torch.rand_like(self.last_contacts)
            latency = torch.where(latency_samples < self.latency, 1, 0)  # with 0.25 probability, the signal is lagged
            self.last_contacts = self.last_contacts * latency + contacts * (1 - latency)
            
            mask = torch.rand_like(self.last_contacts)
            mask = torch.where(mask < self.sensor_noise, 0.0, 1.0)

            sensed_contacts = torch.where(self.last_contacts > 0.1, mask * self.last_contacts, self.last_contacts)
            self.sensed_contacts = sensed_contacts
            if self.viewer:
                self.debug_contacts = sensed_contacts.detach().cpu().numpy()
        
        # obj ends
        prev_obj_ends = self.obj_ends_history[:, 1:].clone()
        pencil_ends = [
            [0, 0, -(self.pen_length/2) * self.obj_scale],
            [0, 0, (self.pen_length/2) * self.obj_scale]
        ]
        pencil_end_1 = self.object_pos + quat_apply(
            self.object_rot, to_torch(pencil_ends[0], device=self.device)[None].repeat(self.num_envs, 1)
        ) - self.root_state_tensor[self.hand_indices, :3]
        pencil_end_2 = self.object_pos + quat_apply(
            self.object_rot, to_torch(pencil_ends[1], device=self.device)[None].repeat(self.num_envs, 1)
        ) - self.root_state_tensor[self.hand_indices, :3]
        pencil_end_1 += (torch.rand(pencil_end_1.shape[0], 3).to(self.device) - 0.5) * (self.pen_radius*2) 
        pencil_end_2 += (torch.rand(pencil_end_2.shape[0], 3).to(self.device) - 0.5) * (self.pen_radius*2) 

        cur_obj_ends = torch.cat([pencil_end_1, pencil_end_2], dim=-1).unsqueeze(1)
        self.obj_ends_history[:] = torch.cat([prev_obj_ends, cur_obj_ends], dim=1)

        t_buf = (self.obs_buf_lag_history[:, -3:,:self.obs_buf.shape[1]//3].reshape(self.num_envs, -1)).clone()
        self.obs_buf[:, :t_buf.shape[1]] = t_buf  # [1, 96]
        
        # deal with normal observation, do sliding windows
        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
        cur_obs_buf = noisy_joint_pos.clone().unsqueeze(1)  # [1, 1, 16]
        cur_tar_buf = self.cur_targets[:, None]  # [1, 1, 16]
        cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)  # [1, 1, 32]
        if self.config['env']['privInfo']['enable_tactile']:
            cur_obs_buf = torch.cat([cur_obs_buf, sensed_contacts.unsqueeze(1)], dim=-1) # [1, 1, 32+12+32]
            cur_obs_buf = torch.cat([cur_obs_buf, self.fingertip_pos.clone().unsqueeze(1)], dim=-1) # [1, 1, 32+12]

        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)  # [1, 80, 16+16+32+12]

        # refill the initialized buffers
        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 0:16] = self.init_pose_buf[at_reset_env_ids].unsqueeze(1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 16:32] = self.init_pose_buf[at_reset_env_ids].unsqueeze(1)
        self.obj_ends_history[at_reset_env_ids, :, :] = cur_obj_ends[at_reset_env_ids]
        if self.config['env']['privInfo']['enable_tactile']:
            self.obs_buf_lag_history[at_reset_env_ids, :, 32:64] = torch.zeros((len(at_reset_env_ids),80,32),device=self.device)
            self.obs_buf_lag_history[at_reset_env_ids, :, 64:76] = self.fingertip_pos[at_reset_env_ids].unsqueeze(1)
        
        # velocity reset
        self.obj_linvel_at_cf[at_reset_env_ids] = self.object_linvel[at_reset_env_ids]
        self.obj_angvel_at_cf[at_reset_env_ids] = self.object_angvel[at_reset_env_ids]
        self.ft_linvel_at_cf[at_reset_env_ids] = self.fingertip_linvel[at_reset_env_ids]
        self.ft_angvel_at_cf[at_reset_env_ids] = self.fingertip_angvel[at_reset_env_ids]

        self.at_reset_buf[at_reset_env_ids] = 0
        rand_rpy = torch.normal(0, self.noisy_rpy_scale, size=(self.num_envs, 3), device=self.device, dtype=torch.float)
        rand_quat = quat_from_euler_xyz(rand_rpy[:, 0], rand_rpy[:, 1], rand_rpy[:, 2])
        noisy_quat = quat_mul(rand_quat, self.object_rot)
        noisy_position = torch.normal(0, self.noisy_pos_scale, size=(self.num_envs, 3), device=self.device, dtype=torch.float) + self.object_pos
        self.noisy_quaternion_buf[at_reset_env_ids, :, :4] = noisy_quat[at_reset_env_ids].unsqueeze(1)
        self.noisy_quaternion_buf[at_reset_env_ids, :, 4:] = noisy_position[at_reset_env_ids].unsqueeze(1)
        self.noisy_quaternion_buf[:] = torch.cat([
            self.noisy_quaternion_buf[:, 1:].clone(),
            torch.cat([noisy_quat.unsqueeze(1), noisy_position.unsqueeze(1)], dim=-1)
        ], dim=1)

        self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.prop_hist_len:, :32]  # [1, 30, 32]
        if self.config['env']['privInfo']['enable_tactile']:
            self.tactile_hist_buf[:] = self.obs_buf_lag_history[:, -self.prop_hist_len:, 32:64]
        self._update_priv_buf(env_id=range(self.num_envs), name='obj_position', value=self.object_pos.clone())
        self._update_priv_buf(env_id=range(self.num_envs), name='obj_orientation', value=self.object_rot.clone())
        self._update_priv_buf(env_id=range(self.num_envs), name='obj_linvel', value=self.obj_linvel_at_cf.clone())
        self._update_priv_buf(env_id=range(self.num_envs), name='obj_angvel', value=self.obj_angvel_at_cf.clone())
        self._update_priv_buf(env_id=range(self.num_envs), name='fingertip_position', value=self.fingertip_pos.clone())
        self._update_priv_buf(env_id=range(self.num_envs), name='fingertip_orientation', value=self.fingertip_orientation.clone())
        self._update_priv_buf(env_id=range(self.num_envs), name='fingertip_linvel', value=self.ft_linvel_at_cf.clone())
        self._update_priv_buf(env_id=range(self.num_envs), name='fingertip_angvel', value=self.ft_angvel_at_cf.clone())
        if self.config['env']['privInfo']['enable_tactile']:
            self._update_priv_buf(env_id=range(self.num_envs), name='tactile', value=self.sensed_contacts.clone())
            
        # update critic observation
        # note: the critic will receive normal observation, privileged info, and critic info
        # deprecated
        self.critic_info_buf[:, 0:4] = self.object_rot
        self.critic_info_buf[:, 4:7] = self.obj_linvel_at_cf
        self.critic_info_buf[:, 7:10] = self.obj_angvel_at_cf
        # 4, 8, 12, 16 are fingertip indexes for rigid body states
        fingertip_states = self.rigid_body_states[:, self.fingertip_handles].clone()
        self.critic_info_buf[:, 10:10 + 13 * 4] = fingertip_states.reshape(self.num_envs, -1)
        if self.point_cloud_sampled_dim > 0:
            # for collecting bc data
            self.point_cloud_buf[:, :self.point_cloud_sampled_dim] = quat_apply(
                self.object_rot[:, None].repeat(1, self.point_cloud_sampled_dim, 1), self.obj_point_clouds
            ) + self.object_pos[:, None]  # [1, 100, 3]
    
    def _get_reward_scale_by_name(self, name):
        env_steps = (self.gym.get_frame_count(self.sim) * len(self.envs))
        agent_steps = env_steps // self.control_freq_inv
        init_scale, final_scale, curr_start, curr_end = self.reward_scale_dict[name]
        if curr_end > 0:
            curr_progress = (agent_steps - curr_start) / (curr_end - curr_start)
            curr_progress = min(max(curr_progress, 0), 1)
            # discretize to [0, 0.05, 1.0] instead of continuous value
            # during batch collection, avoid reward confusion
            curr_progress = round(curr_progress * 20) / 20
        else:
            curr_progress = 1
        if self.evaluate:
            curr_progress = 1
        return init_scale + (final_scale - init_scale) * curr_progress

    def compute_reward(self, actions):
        # pose diff penalty
        pose_diff_penalty = ((self.allegro_hand_dof_pos - self.init_pose_buf) ** 2).sum(-1)
        # work and torque penalty
        # TODO: only consider -1 is incorrect, but need to find the new scale
        torque_penalty = (self.torques[:, -1] ** 2).sum(-1)
        work_penalty = (((torch.abs(self.torques[:, -1]) * torch.abs(self.dof_vel_finite_diff[:, -1])).sum(-1)) ** 2)
        # Compute offset in radians. Radians -> radians / sec
        angdiff = quat_to_axis_angle(quat_mul(self.object_rot, quat_conjugate(self.object_rot_prev)))
        object_angvel = angdiff / (self.control_freq_inv * self.dt)
        vec_dot = (object_angvel * self.rot_axis_buf).sum(-1)
        rotate_reward = torch.clip(vec_dot, max=self.angvel_clip_max, min=self.angvel_clip_min)
        rotate_penalty = torch.where(vec_dot > self.angvel_penalty_threshold, vec_dot - self.angvel_penalty_threshold, 0)
        # linear velocity: use position difference instead of self.object_linvel
        object_linvel = ((self.object_pos - self.object_pos_prev) / (self.control_freq_inv * self.dt)).clone()
        object_linvel_penalty = torch.norm(object_linvel, p=1, dim=-1)
        # TODO: move this to a more appropriate place
        self.obj_angvel_at_cf = object_angvel
        self.obj_linvel_at_cf = object_linvel
        ft_angdiff = quat_to_axis_angle(quat_mul(self.fingertip_orientation.reshape(-1, 4), quat_conjugate(self.ft_rot_prev.reshape(-1, 4)))).reshape(-1, 12)
        self.ft_angvel_at_cf = ft_angdiff / (self.control_freq_inv * self.dt)
        self.ft_linvel_at_cf = ((self.fingertip_pos - self.ft_pos_prev) / (self.control_freq_inv * self.dt))

        if self.point_cloud_sampled_dim > 0:
            point_cloud_z = self.point_cloud_buf[:, :self.point_cloud_sampled_dim, -1]
            z_dist_penalty = point_cloud_z.max(axis=1)[0] - point_cloud_z.min(axis=1)[0]
            z_dist_penalty[z_dist_penalty <= 0.03] = 0
        else:
            z_dist_penalty = to_torch([0], device=self.device)

        # penalize large deviation of cube
        position_penalty = (self.object_pos[:, 0] - 0) ** 2 + (self.object_pos[:, 1] - 0) ** 2 \
            + (self.object_pos[:, 2] - (self.reset_z_threshold + 0.01)) ** 2
        # finger obj deviation penalty
        finger_obj_penalty = ((self.fingertip_pos - self.object_pos.repeat(1, 4)) ** 2).sum(-1)

        self.rew_buf[:] = compute_hand_reward(
            object_linvel_penalty, self._get_reward_scale_by_name('obj_linvel_penalty'),
            rotate_reward, self._get_reward_scale_by_name('rotate_reward'),
            pose_diff_penalty, self._get_reward_scale_by_name('pose_diff_penalty'),
            torque_penalty, self._get_reward_scale_by_name('torque_penalty'),
            work_penalty, self._get_reward_scale_by_name('work_penalty'),
            z_dist_penalty, self._get_reward_scale_by_name('pencil_z_dist_penalty'),
            position_penalty, self._get_reward_scale_by_name('position_penalty'),
            rotate_penalty, self._get_reward_scale_by_name('rotate_penalty')
        )

        self.reset_buf[:] = self.check_termination(self.object_pos)
        self.extras['step_all_reward'] = self.rew_buf.mean()
        self.extras['rotation_reward'] = rotate_reward.mean()
        self.extras['penalty/position'] = position_penalty.mean()
        self.extras['penalty/finger_obj'] = finger_obj_penalty.mean()
        self.extras['object_linvel_penalty'] = object_linvel_penalty.mean()
        self.extras['pose_diff_penalty'] = pose_diff_penalty.mean()
        self.extras['work_done'] = work_penalty.mean()
        self.extras['torques'] = torque_penalty.mean()
        self.extras['roll'] = torch.abs(object_angvel[:, 0]).mean()
        self.extras['pitch'] = torch.abs(object_angvel[:, 1]).mean()
        self.extras['yaw'] = torch.abs(object_angvel[:, 2]).mean()
        self.extras['z_dist_penalty'] = z_dist_penalty.mean()

        if self.evaluate:
            for i in range(len(self.object_type_list)):
                env_ids = torch.where(self.object_type_at_env == i)
                if len(env_ids[0]) > 0:
                    running_mask = 1 - self.eval_done_buf[env_ids]
                    self.stat_sum_rewards[i] += (running_mask * self.rew_buf[env_ids]).sum()
                    self.stat_sum_episode_length[i] += running_mask.sum()
                    self.stat_sum_rotate_rewards[i] += (running_mask * rotate_reward[env_ids]).sum()
                    self.stat_sum_unclip_rotate_rewards[i] += (running_mask * vec_dot[env_ids]).sum()

                    # Update eval_done_buf when evaluating just one object. This will
                    # stop tracking statistics after environment resets.
                    if self.config['env']['object']['evalObjectType'] is not None:
                        flip = running_mask * self.reset_buf[env_ids]
                        self.env_evaluated[i] += flip.sum()
                        self.eval_done_buf[env_ids] += flip

                    info = f'Progress: {self.evaluate_iter} / {self.max_episode_length}'
                    tprint(info)
            self.evaluate_iter += 1

    def post_physics_step(self):
        self.progress_buf += 1
        self.reset_buf[:] = 0
        self._refresh_gym()
        # cur* but need for reward is here
        self.compute_reward(self.actions)

        # calibration
        # self.reset_buf[:] = 0

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.compute_observations()

        self.debug_viz = False
        if self.viewer and self.config['env']['privInfo']['enable_tactile']:
            for env in range(len(self.envs)):
                for i, contact_idx in enumerate(list(self.sensor_handle_indices)):

                    if self.debug_contacts[env, i] > 0.0:
                        self.gym.set_rigid_body_color(self.envs[env], self.hand_actors[env],
                                                      contact_idx, gymapi.MESH_VISUAL_AND_COLLISION,
                                                      gymapi.Vec3(0.0, 1.0, 0.0))
                    else:
                        self.gym.set_rigid_body_color(self.envs[env], self.hand_actors[env],
                                                      contact_idx, gymapi.MESH_VISUAL_AND_COLLISION,
                                                      gymapi.Vec3(1.0, 0.0, 0.0))
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = self.prev_targets + self.action_scale * self.actions
        # targets = self.actions.clone()
        self.cur_targets[:] = tensor_clamp(targets, self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)
        # get prev* buffer here
        self.prev_targets[:] = self.cur_targets
        self.object_rot_prev[:] = self.object_rot
        self.object_pos_prev[:] = self.object_pos
        self.ft_rot_prev[:] = self.fingertip_orientation
        self.ft_pos_prev[:] = self.fingertip_pos
        self.dof_vel_prev[:] = self.dof_vel_finite_diff

    def reset(self):
        super().reset()
        self.obs_dict['priv_info'] = self.priv_info_buf.to(self.rl_device)
        self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.rl_device)
        self.obs_dict['tactile_hist'] = self.tactile_hist_buf.to(self.rl_device)
        self.obs_dict['noisy_quaternion'] = self.noisy_quaternion_buf.to(self.rl_device)
        # observation buffer for critic
        self.obs_dict['critic_info'] = self.critic_info_buf.to(self.rl_device)
        self.obs_dict['point_cloud_info'] = self.point_cloud_buf.to(self.rl_device)
        self.obs_dict['rot_axis_buf'] = self.rot_axis_buf.to(self.rl_device)
        if self.enable_obj_ends:
            self.obs_dict['obj_ends'] = self.obj_ends_history.to(self.rl_device)
        return self.obs_dict

    def step(self, actions, extrin_record: Optional[torch.Tensor] = None):
        # Save extrinsics if evaluating on just one object.
        if extrin_record is not None and self.config['env']['object']['evalObjectType'] is not None:
            # Put a (z vectors, is done) tuple into the log.
            self.extrin_log.append(
                (extrin_record.detach().cpu().numpy().copy(), self.eval_done_buf.detach().cpu().numpy().copy())
            )

        super().step(actions)
        self.obs_dict['priv_info'] = self.priv_info_buf.to(self.rl_device)
        # stage 2 buffer
        self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.rl_device)
        self.obs_dict['tactile_hist'] = self.tactile_hist_buf.to(self.rl_device)
        self.obs_dict['noisy_quaternion'] = self.noisy_quaternion_buf.to(self.rl_device)
        # observation buffer for critic
        self.obs_dict['critic_info'] = self.critic_info_buf.to(self.rl_device)
        self.obs_dict['point_cloud_info'] = self.point_cloud_buf.to(self.rl_device)
        self.obs_dict['rot_axis_buf'] = self.rot_axis_buf.to(self.rl_device)
        if self.enable_obj_ends:
            self.obs_dict['obj_ends'] = self.obj_ends_history.to(self.rl_device)
        return self.obs_dict, self.rew_buf, self.reset_buf, self.extras

    def capture_frame(self) -> np.ndarray:
        assert self.enable_camera_sensors  # camera sensors should be enabled
        assert self.vid_record_tensor is not None
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        frame = self.vid_record_tensor.cpu().numpy()
        self.gym.end_access_image_tensors(self.sim)

        return frame

    def update_low_level_control(self, step_id):
        previous_dof_pos = self.allegro_hand_dof_pos.clone()
        self._refresh_gym()
        random_action_noise_t = torch.normal(0, self.random_action_noise_t_scale, size=self.allegro_hand_dof_pos.shape, device=self.device, dtype=torch.float)
        noise_action = self.cur_targets + self.random_action_noise_e + random_action_noise_t
        if self.torque_control:
            dof_pos = self.allegro_hand_dof_pos
            dof_vel = (dof_pos - previous_dof_pos) / self.dt
            self.dof_vel_finite_diff[:, step_id] = dof_vel.clone()
            torques = self.p_gain * (noise_action - dof_pos) - self.d_gain * dof_vel
            torques = torch.clip(torques, -self.torque_limit, self.torque_limit).clone()
            self.torques[:, step_id] = torques
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
        else:
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(noise_action))

    def update_rigid_body_force(self):
        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)
            # apply new forces
            obj_mass = [self.gym.get_actor_rigid_body_properties(env, self.gym.find_actor_handle(env, 'object'))[0].mass for env in self.envs]
            obj_mass = to_torch(obj_mass, device=self.device)
            prob = self.random_force_prob_scalar
            force_indices = (torch.less(torch.rand(self.num_envs, device=self.device), prob)).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape,
                device=self.device) * obj_mass[force_indices, None] * self.force_scale
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.ENV_SPACE)

    def check_termination(self, object_pos):
        term_by_max_eps = torch.greater_equal(self.progress_buf, self.max_episode_length)
        # default option
        reset_z = torch.less(object_pos[:, -1], self.reset_z_threshold)
        resets = reset_z
        resets = torch.logical_or(resets, term_by_max_eps)

        if self.canonical_pose_category == 'pencil':
            pencil_ends = [
                [0, 0, -(self.pen_length/2) * self.obj_scale],
                [0, 0, (self.pen_length/2) * self.obj_scale]
            ]
            pencil_end_1 = self.object_pos + quat_apply(self.object_rot, to_torch(pencil_ends[0], device=self.device)[None].repeat(self.num_envs, 1))
            pencil_end_2 = self.object_pos + quat_apply(self.object_rot, to_torch(pencil_ends[1], device=self.device)[None].repeat(self.num_envs, 1))
            pencil_z_min = torch.min(pencil_end_1, pencil_end_2)[:, -1]
            pencil_z_max = torch.max(pencil_end_1, pencil_end_2)[:, -1]
            pencil_fall = torch.logical_or(pencil_z_min < 0.56, pencil_z_max > 0.71)
            resets = torch.logical_or(resets, pencil_fall)

        return resets

    def _refresh_gym(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]
        self.fingertip_states = self.rigid_body_states[:, self.fingertip_handles]
        self.fingertip_pos = self.fingertip_states[:, :, :3].reshape(self.num_envs, -1)
        self.fingertip_orientation = self.fingertip_states[:, :, 3:7].reshape(self.num_envs, -1)
        self.fingertip_linvel = self.fingertip_states[:, :, 7:10].reshape(self.num_envs, -1)
        self.fingertip_angvel = self.fingertip_states[:, :, 10:13].reshape(self.num_envs, -1)

    def _setup_domain_rand_config(self, rand_config):
        self.randomize_mass = rand_config['randomizeMass']
        self.randomize_mass_lower = rand_config['randomizeMassLower']
        self.randomize_mass_upper = rand_config['randomizeMassUpper']
        self.randomize_com = rand_config['randomizeCOM']
        self.randomize_com_lower = rand_config['randomizeCOMLower']
        self.randomize_com_upper = rand_config['randomizeCOMUpper']
        self.randomize_friction = rand_config['randomizeFriction']
        self.randomize_friction_lower = rand_config['randomizeFrictionLower']
        self.randomize_friction_upper = rand_config['randomizeFrictionUpper']
        self.randomize_scale = rand_config['randomizeScale']
        self.randomize_hand_scale = rand_config['randomize_hand_scale']
        self.scale_list_init = rand_config['scaleListInit']
        self.randomize_scale_list = rand_config['randomizeScaleList']
        self.randomize_scale_lower = rand_config['randomizeScaleLower']
        self.randomize_scale_upper = rand_config['randomizeScaleUpper']
        self.randomize_pd_gains = rand_config['randomizePDGains']
        self.randomize_p_gain_lower = rand_config['randomizePGainLower']
        self.randomize_p_gain_upper = rand_config['randomizePGainUpper']
        self.randomize_d_gain_lower = rand_config['randomizeDGainLower']
        self.randomize_d_gain_upper = rand_config['randomizeDGainUpper']
        self.random_obs_noise_e_scale = rand_config['obs_noise_e_scale']
        self.random_obs_noise_t_scale = rand_config['obs_noise_t_scale']
        self.random_pose_noise = rand_config['pose_noise_scale']
        self.random_action_noise_e_scale = rand_config['action_noise_e_scale']
        self.random_action_noise_t_scale = rand_config['action_noise_t_scale']
        # stage 2 specific
        self.noisy_rpy_scale = rand_config['noisy_rpy_scale']
        self.noisy_pos_scale = rand_config['noisy_pos_scale']

        self.sensor_thresh = 1.0
        self.sensor_noise = 0.1
        self.latency = 0.2

    def _setup_priv_option_config(self, p_config):
        self.enable_priv_obj_position = p_config['enableObjPos']
        self.enable_priv_obj_mass = p_config['enableObjMass']
        self.enable_priv_obj_scale = p_config['enableObjScale']
        self.enable_priv_obj_com = p_config['enableObjCOM']
        self.enable_priv_obj_friction = p_config['enableObjFriction']
        self.contact_input_dim = p_config['contact_input_dim']
        self.contact_form = p_config['contact_form']
        self.contact_input = p_config['contact_input']
        self.contact_binarize_threshold = p_config['contact_binarize_threshold']
        self.enable_priv_obj_orientation = p_config['enable_obj_orientation']
        self.enable_priv_obj_linvel = p_config['enable_obj_linvel']
        self.enable_priv_obj_angvel = p_config['enable_obj_angvel']
        self.enable_priv_fingertip_position = p_config['enable_ft_pos']
        self.enable_priv_fingertip_orientation = p_config['enable_ft_orientation']
        self.enable_priv_fingertip_linvel = p_config['enable_ft_linvel']
        self.enable_priv_fingertip_angvel = p_config['enable_ft_angvel']
        self.enable_priv_hand_scale = p_config['enable_hand_scale']
        self.enable_priv_obj_restitution = p_config['enable_obj_restitution']
        self.enable_priv_tactile = p_config['enable_tactile']

        hand_asset_file = self.config['env']['asset']['handAsset']
        if hand_asset_file == "assets/round_tip/allegro_hand_right_fsr_round_dense.urdf":
            self.num_contacts = 5 * 4 + 12
        else:
            self.num_contacts = 0
        if not self.config['env']['privInfo']['enable_tactile']:
            self.num_contacts = 0

        self.priv_info_dict = {
            'obj_position': (0, 3),
            'obj_scale': (3, 4),
            'obj_mass': (4, 5),
            'obj_friction': (5, 6),
            'obj_com': (6, 9),
        }
        start_index = 9

        priv_dims = OrderedDict()
        priv_dims['obj_orientation'] = 4
        priv_dims['obj_linvel'] = 3
        priv_dims['obj_angvel'] = 3
        priv_dims['fingertip_position'] = 3 * 4
        priv_dims['fingertip_orientation'] = 4 * 4
        priv_dims['fingertip_linvel'] = 4 * 3
        priv_dims['fingertip_angvel'] = 4 * 3
        priv_dims['hand_scale'] = 1
        priv_dims['obj_restitution'] = 1
        priv_dims['tactile'] = self.num_contacts
        for name, dim in priv_dims.items():
            if eval(f'self.enable_priv_{name}'):
                self.priv_info_dict[name] = (start_index, start_index + dim)
                start_index += dim

    def _update_priv_buf(self, env_id, name, value):
        # normalize to -1, 1
        if eval(f'self.enable_priv_{name}'):
            s, e = self.priv_info_dict[name]
            if type(value) is list:
                value = to_torch(value, dtype=torch.float, device=self.device)
            self.priv_info_buf[env_id, s:e] = value

    def _setup_object_info(self, o_config):
        self.object_type = o_config['type']
        raw_prob = o_config['sampleProb']
        assert (sum(raw_prob) == 1)

        primitive_list = self.object_type.split('+')
        print('---- Primitive List ----')
        print(primitive_list)
        self.object_type_prob = []
        self.object_type_list = []
        self.asset_files_dict = {
            'simple_tennis_ball': 'assets/ball.urdf',
            'simple_cube': 'assets/cube.urdf',
            'simple_cylin4cube': 'assets/cylinder4cube.urdf',
        }
        for p_id, prim in enumerate(primitive_list):
            if 'cuboid' in prim:
                subset_name = self.object_type.split('_')[-1]
                cuboids = sorted(glob(f'assets/cuboid/{subset_name}/*.urdf'))
                cuboid_list = [f'cuboid_{i}' for i in range(len(cuboids))]
                self.object_type_list += cuboid_list
                for i, name in enumerate(cuboids):
                    self.asset_files_dict[f'cuboid_{i}'] = name.replace('../assets/', '')
                self.object_type_prob += [raw_prob[p_id] / len(cuboid_list) for _ in cuboid_list]
            elif 'cylinder' in prim:
                subset_name = self.object_type.split('_')[-1]
                cylinders = sorted(glob(f'assets/cylinder/{subset_name}/*.urdf'))
                cylinder_list = [f'cylinder_{i}' for i in range(len(cylinders))]
                self.object_type_list += cylinder_list
                for i, name in enumerate(cylinders):
                    self.asset_files_dict[f'cylinder_{i}'] = name.replace('../assets/', '')
                self.object_type_prob += [raw_prob[p_id] / len(cylinder_list) for _ in cylinder_list]
            else:
                self.object_type_list += [prim]
                self.object_type_prob += [raw_prob[p_id]]
        print('---- Object List ----')
        print(f'using {len(self.object_type_list)} training objects')
        assert (len(self.object_type_list) == len(self.object_type_prob))

    def _allocate_task_buffer(self, num_envs):
        # extra buffers for observe randomized params
        self.prop_hist_len = self.config['env']['hora']['propHistoryLen']
        self.priv_info_dim = max([v[1] for k, v in self.priv_info_dict.items()])
        self.critic_obs_dim = self.config['env']['hora']['critic_obs_dim']
        self.point_cloud_sampled_dim = self.config['env']['hora']['point_cloud_sampled_dim']
        self.point_cloud_buffer_dim = self.point_cloud_sampled_dim
        self.priv_info_buf = torch.zeros((num_envs, self.priv_info_dim), device=self.device, dtype=torch.float)
        self.critic_info_buf = torch.zeros((num_envs, self.critic_obs_dim), device=self.device, dtype=torch.float)
        # for collecting bc data
        self.point_cloud_buf = torch.zeros((num_envs, self.point_cloud_sampled_dim, 3), device=self.device, dtype=torch.float)
        # fixed noise per-episode, for different hardware have different this value
        self.random_obs_noise_e = torch.zeros((num_envs, self.config['env']['numActions']), device=self.device, dtype=torch.float)
        self.random_action_noise_e = torch.zeros((num_envs, self.config['env']['numActions']), device=self.device, dtype=torch.float)
        # ---- stage 2 buffers
        # stage 2 related buffers
        self.proprio_hist_buf = torch.zeros((num_envs, self.prop_hist_len, 32), device=self.device, dtype=torch.float)
        self.tactile_hist_buf = torch.zeros((num_envs, self.prop_hist_len, 32), device=self.device, dtype=torch.float)
        # a bit unintuitive: first 4 is quaternion and last 3 is position, due to development order
        self.noisy_quaternion_buf = torch.zeros((num_envs, self.prop_hist_len, 7), device=self.device, dtype=torch.float)

    def _setup_reward_config(self, r_config):
        # the list
        self.reward_scale_dict = {}
        for k, v in r_config.items():
            if 'scale' in k:
                if type(v) is not omegaconf.listconfig.ListConfig:
                    v = [v, v, 0, 0]
                else:
                    assert len(v) == 4
                self.reward_scale_dict[k.replace('_scale', '')] = v
        self.angvel_clip_min = r_config['angvelClipMin']
        self.angvel_clip_max = r_config['angvelClipMax']
        self.angvel_penalty_threshold = r_config['angvelPenaltyThres']

    def _create_object_asset(self):
        # object file to asset
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
        hand_asset_file = self.config['env']['asset']['handAsset']
        # load hand asset
        hand_asset_options = gymapi.AssetOptions()
        hand_asset_options.flip_visual_attachments = False
        hand_asset_options.fix_base_link = True
        hand_asset_options.collapse_fixed_joints = False
        hand_asset_options.convex_decomposition_from_submeshes = True
        hand_asset_options.disable_gravity = True
        hand_asset_options.thickness = 0.001
        hand_asset_options.angular_damping = 0.01

        if self.torque_control:
            hand_asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)
        else:
            hand_asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
        self.hand_asset = self.gym.load_asset(self.sim, asset_root, hand_asset_file, hand_asset_options)
        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(self.hand_asset, name) for name in
                                  ['link_3.0_tip', 'link_15.0_tip', 'link_7.0_tip', 'link_11.0_tip']]
        self.contact_sensor_names = ["link_1.0_fsr", "link_2.0_fsr", "link_5.0_fsr",
                                     "link_6.0_fsr", "link_9.0_fsr", "link_10.0_fsr",
                                     "link_14.0_fsr", "link_15.0_fsr", "link_0.0_fsr", 
                                     "link_4.0_fsr", "link_8.0_fsr", "link_13.0_fsr"]
        for tip_name in ['3.0', '15.0', '7.0', '11.0']:
            if hand_asset_file == "assets/round_tip/allegro_hand_right_fsr_round_dense.urdf":
                tip_fsr_range = [2, 5, 8, 11, 13]
            else:
                tip_fsr_range = []
            for i in tip_fsr_range:
                self.contact_sensor_names.append("link_{}_tip_fsr_{}".format(tip_name, str(i)))

        # load object asset
        self.object_asset_list = []
        self.asset_point_clouds = []
        for object_type in self.object_type_list:
            object_asset_file = self.asset_files_dict[object_type]
            object_asset_options = gymapi.AssetOptions()
            # If we've specified a specific eval object, we only need to load that object.
            eval_object_type = self.config['env']['object']['evalObjectType']
            if eval_object_type is not None and object_type != eval_object_type:
                self.object_asset_list.append(None)
                self.asset_point_clouds.append(None)
                continue

            object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
            self.object_asset_list.append(object_asset)
            if 'cylin4cube' in object_type and self.point_cloud_sampled_dim > 0:
                pcs = sample_cylinder(1) * 0.08
                pcs[:, :2] *= 1.2
                self.asset_point_clouds.append(pcs)
            else:
                if 'cylinder' in object_type:
                    # dim 0 is cylinder height, 1 = 0.08m
                    # dim 1,2 are cylinder diameter, 1 = 0.08m [radius 4cm] 
                    size_info = np.load(os.path.join(asset_root, object_asset_file.replace('.urdf', '.npy')))[0]
                    self.pen_radius = size_info[1]
                    self.pen_length = size_info[0] * (size_info[1] * 2)
                    print("loading", os.path.join(asset_root, object_asset_file.replace('.urdf', '.npy')), size_info)
                    if self.point_cloud_sampled_dim > 0:
                        self.asset_point_clouds.append(sample_cylinder(size_info[0]) * self.pen_radius * 2)
                elif ('cube' in object_type or 'cuboid' in object_type) and self.point_cloud_sampled_dim > 0:
                    size_info = np.load(os.path.join(asset_root, object_asset_file.replace('.urdf', '.npy')))[0]
                    self.asset_point_clouds.append(sample_cuboid(size_info[0] * 0.08, size_info[1] * 0.08, size_info[2] * 0.08))

        assert any([x is not None for x in self.object_asset_list])
        # assert any([x is not None for x in self.asset_point_clouds])

    def _parse_hand_dof_props(self):
        self.num_allegro_hand_dofs = self.gym.get_asset_dof_count(self.hand_asset)
        allegro_hand_dof_props = self.gym.get_asset_dof_properties(self.hand_asset)

        self.allegro_hand_dof_lower_limits = []
        self.allegro_hand_dof_upper_limits = []

        for i in range(self.num_allegro_hand_dofs):
            # another option, just do it for now, parse directly from Nvidia's Calibrated Value
            # avoid frequently or adding another URDF
            allegro_hand_dof_lower_limits = [
                -0.5585, -0.27924, -0.27924, -0.27924, 0.27924, -0.331603, -0.27924, -0.27924,
                -0.5585, -0.27924, -0.27924, -0.27924, -0.5585, -0.27924, -0.27924, -0.27924,
            ]
            allegro_hand_dof_upper_limits = [
                0.5585, 1.727825, 1.727825, 1.727825, 1.57075, 1.1518833, 1.727825, 1.76273055,
                0.5585, 1.727825, 1.727825, 1.727825, 0.5585, 1.727825, 1.727825, 1.727825,
            ]
            allegro_hand_dof_props['lower'][i] = allegro_hand_dof_lower_limits[i]
            allegro_hand_dof_props['upper'][i] = allegro_hand_dof_upper_limits[i]
            
            self.allegro_hand_dof_lower_limits.append(allegro_hand_dof_props['lower'][i])
            self.allegro_hand_dof_upper_limits.append(allegro_hand_dof_props['upper'][i])
            allegro_hand_dof_props['effort'][i] = self.torque_limit
            if self.torque_control:
                allegro_hand_dof_props['stiffness'][i] = 0.
                allegro_hand_dof_props['damping'][i] = 0.
                allegro_hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
            else:
                allegro_hand_dof_props['stiffness'][i] = self.config['env']['controller']['pgain']
                allegro_hand_dof_props['damping'][i] = self.config['env']['controller']['dgain']
            allegro_hand_dof_props['friction'][i] = 0.01
            allegro_hand_dof_props['armature'][i] = 0.001

        self.allegro_hand_dof_lower_limits = to_torch(self.allegro_hand_dof_lower_limits, device=self.device)
        self.allegro_hand_dof_upper_limits = to_torch(self.allegro_hand_dof_upper_limits, device=self.device)
        return allegro_hand_dof_props

    def _init_object_pose(self):
        allegro_hand_start_pose = gymapi.Transform()
        allegro_hand_start_pose.p = gymapi.Vec3(0, 0, 0.5)
        allegro_hand_start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), -np.pi / 2) * \
                                   gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.pi / 2)
        pose_dx, pose_dy, pose_dz = 0.00, -0.04, 0.15
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = allegro_hand_start_pose.p.x
        object_start_pose.p.x = allegro_hand_start_pose.p.x + pose_dx
        object_start_pose.p.y = allegro_hand_start_pose.p.y + pose_dy
        object_start_pose.p.z = allegro_hand_start_pose.p.z + pose_dz

        object_start_pose.p.y = allegro_hand_start_pose.p.y - 0.01
        # TODO: this weird thing is an unknown issue
        if self.save_init_pose:
            object_start_pose.p.z = (self.reset_z_threshold + 0.015)
        else:
            object_start_pose.p.z = (self.reset_z_threshold + 0.005)
        return allegro_hand_start_pose, object_start_pose


def compute_hand_reward(
    object_linvel_penalty, object_linvel_penalty_scale: float,
    rotate_reward, rotate_reward_scale: float,
    pose_diff_penalty, pose_diff_penalty_scale: float,
    torque_penalty, torque_pscale: float,
    work_penalty, work_pscale: float,
    z_dist_penalty, z_dist_penalty_scale: float,
    position_penalty, position_penalty_scale: float,
    rotate_penalty, rotate_penalty_scale: float
):
    reward = rotate_reward_scale * rotate_reward
    reward = reward + object_linvel_penalty * object_linvel_penalty_scale
    reward = reward + pose_diff_penalty * pose_diff_penalty_scale
    reward = reward + torque_penalty * torque_pscale
    reward = reward + work_penalty * work_pscale
    reward = reward + z_dist_penalty * z_dist_penalty_scale
    reward = reward + position_penalty * position_penalty_scale
    reward = reward + rotate_penalty * rotate_penalty_scale
    return reward


def quat_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Adapted from PyTorch3D:
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#quaternion_to_axis_angle

    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., :3], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., 3:])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., :3] / sin_half_angles_over_angles
