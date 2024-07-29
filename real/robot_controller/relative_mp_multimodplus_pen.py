import sys
# sys.path.append("/media/binghao/DATA/new_rl_pipeline/rl_pipeline")

import torch
import copy
import numpy as np

import rospy
import time

from model.models import ActorCritic
from model.running_mean_std import RunningMeanStd
from model.algos_torch import torch_ext
from real_robot.robot import *
from real_robot.utils import _action_hora2allegro,_obs_allegro2hora


class NNRelativeMLPControllerMP:
	def __init__(self, dof_lower, dof_upper, num_actors=1, scale=0.5, stack=3, device='cuda', 
			  sam=True, proprio_mode=False, enable_obj_ends=False, input_mode='proprio-ends'):

		self.num_actors = num_actors
		self.action_scale = 0.04167
		self.actions_num = 16
		self.control_freq = 18
		self.device = device
		self.sam = sam 
		self.visualize = False
		self.proprio_mode = proprio_mode
		self.enable_obj_ends = enable_obj_ends
		self.input_mode = input_mode
		self.robot = RealRobot(self.sam)
        # --------------------------------
        # Depth Image Setting
		self.transformer = True
        # self.segd_camera = RealsenseSegD()
		self.proprio_len = 30
		obs_shape = (96,)
		 
		net_config = {
			'actor_units': [512, 256, 128],
			'priv_mlp_units': [256, 128, 8],
            'actions_num': 16,
            'input_shape': obs_shape,
            'priv_info':  False,
            'proprio_adapt': False,
            'priv_info_dim': 61,
            'asymm_actor_critic': False,
            'critic_info_dim': 100,
            'point_cloud_sampled_dim': 0,
            'point_mlp_units': [32, 32, 32],
			'use_point_transformer': False,
            'use_fine_contact': False,
            'contact_mlp_units': [32, 32, 32],
			'student': True,
			'multi_axis': False,
			'use_point_cloud_info': False,
			'input_mode': self.input_mode,
			'proprio_mode': self.proprio_mode,
			'proprio_len': self.proprio_len
        }
		
		self.model = ActorCritic(net_config)
		self.model.eval()
		self.model.to(self.device)

		self.running_mean_std = RunningMeanStd((96,)).to(self.device)
		self.sa_mean_std = RunningMeanStd((3, 32)).to(self.device)
		self.sa_mean_std.eval()
		#self.last_action = torch.zeros(num_actors, 22).to(self.device))
		self.last_action = torch.zeros(num_actors, 16).to(self.device)
		# self.prev_target = torch.zeros(num_actors, 22).to(self.device))
		self.prev_target = torch.zeros(num_actors, 16).to(self.device)
		#self.initial_target = torch.zeros(22).to(self.device))
		self.initial_target = torch.zeros(16).to(self.device)

		self.states = None
		self.dof_lower = torch.from_numpy(dof_lower).to(self.device)
		self.dof_upper = torch.from_numpy(dof_upper).to(self.device)
		self.all_commands = torch.tensor([[1.0, 0.0, 0.0],
										  [0.0, -1.0, 0.0],
										  [0.0, 0.0, 1.0]]).float()
		self.current_cmd = 0
		self.num_supported_cmd = 3

		self.need_init = 0
		# self.last_action = torch.zeros(num_actors, 22).to(self.device))
		self.last_action = torch.zeros(num_actors, 16).to(self.device)
		self.m_scale = scale

		# self.single_step_obs_dim = 85
		self.single_step_obs_dim = 64
		self.stack = stack
		self.history = torch.zeros(num_actors, self.single_step_obs_dim * self.stack).float()

		###########Initial the Tracker###########
		if self.visualize:
			self.pcd_vis = o3d.geometry.PointCloud()
			self.obj_end = o3d.geometry.PointCloud()
			self.aug_pcd_vis = o3d.geometry.PointCloud()
			self.vis = o3d.visualization.Visualizer()
			self.vis.create_window()
			ctr = self.vis.get_view_control()
			ctr.set_lookat([0,0,0])
			self.coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
			
	def pc_visualization(self, pen_pc,obj_end_pc, aug_pc=None,initialize = False):
		self.pcd_vis.points = pen_pc
		self.pcd_vis.paint_uniform_color([0,0,1])
		self.obj_end.points = obj_end_pc
		self.obj_end.paint_uniform_color([0,1,0])
		if aug_pc is not None:
			self.aug_pcd_vis.points = aug_pc
			self.aug_pcd_vis.paint_uniform_color([1,0,0])
		# if initialize:
		# 	self.vis.add_geometry(self.pcd_vis)
		# 	self.vis.add_geometry(self.coordinate)
		# 	if aug_pc is not None:
		# 		self.vis.add_geometry(self.aug_pcd_vis)
		
		# self.vis.update_geometry(self.pcd_vis)
		# if aug_pc is not None:
		# 		self.vis.add_geometry(self.aug_pcd_vis)
		o3d.visualization.draw_geometries([self.pcd_vis, self.obj_end, self.aug_pcd_vis, self.coordinate])
		# self.vis.poll_events()
		# self.vis.update_renderer()
		

	def unnormalize(self, qpos):
		return (0.5 * (qpos + 1.0) * (self.dof_upper - self.dof_lower) + self.dof_lower)

	def normalize(self, qpos):
		return (2.0 * qpos - self.dof_upper - self.dof_lower) / (self.dof_upper - self.dof_lower)

	def reset(self):
		# self.prev_target = self.initial_target.reshape(-1, 22).repeat(self.num_actors, 1)
		self.prev_target = self.initial_target.reshape(-1, 16).repeat(self.num_actors, 1)
		self.history = torch.zeros(self.num_actors, self.single_step_obs_dim * self.stack).float()
		self.need_init = 1
		# self.reset_rnn()

	def set_initial_target(self, target):
		# target: np.array, absolute qpos.
		self.initial_target = torch.from_numpy(target).float().to(self.device)

	def scale(self, actions):
		return (0.5 * (actions + 1.0) * (self.dof_upper - self.dof_lower) + self.dof_lower)

	def _preproc_obs(self, obs_batch):
		if type(obs_batch) is dict:
			obs_batch = copy.copy(obs_batch)
			for k, v in obs_batch.items():
				if v.dtype == torch.uint8:
					obs_batch[k] = v.float() / 255.0
				else:
					obs_batch[k] = v
		else:
			if obs_batch.dtype == torch.uint8:
				obs_batch = obs_batch.float() / 255.0
		return obs_batch

	def set_command(self, command_id):
		'''

		:param command_id: an integer,
		:return:
		'''
		assert command_id < self.num_supported_cmd, "command is not supported"
		self.current_cmd = command_id

	def get_command(self, batchsize):
		x = self.all_commands[self.current_cmd].reshape(1, -1).repeat(batchsize, 8)
		# print(x.shape)
		return x

	def deploy(self, timesteps, deterministic=True):
		'''
			:param observation: np.array, size = (num_actors, 84, ) dtype=np.float32
			:param deterministic: boolean.
			:return: action: np.array, size = (num_actors, 22, )
		'''
		# Store the History
		count = 0
		#annotate_init_frame
		if self.sam:
			print("SAM is ON")
			self.robot.get_point_cloud_with_Tracking_SAM()


		# hardware deployment buffer
		obs_buf_lag_history = torch.from_numpy(np.zeros((1, self.proprio_len+3, 32 + 32 + 12)).astype(np.float32)).cuda()
		proprio_hist_buf = torch.from_numpy(np.zeros((1, self.proprio_len, 16 * 2)).astype(np.float32)).cuda()
		tactil_hist_buf = torch.from_numpy(np.zeros((1, self.proprio_len, 32)).astype(np.float32)).cuda()
		obj_ends_history = torch.zeros((1, 3, 6), device=self.device, dtype=torch.float32)
		obs_buf = obs_buf_lag_history[:, -3:, :32].clone()

		init_qpos = torch.from_numpy(INIT_QPOS).cuda()
		prev_target = init_qpos
		obs_buf_lag_history = obs_buf_lag_history.to(self.device)
		obs_buf_lag_history[:,:,:16] = init_qpos
		obs_buf_lag_history[:,:,16:32] = init_qpos
		obs_buf_lag_history[:,:,32:64] = torch.zeros(32)

		# Get the tactile values
		tactile_values, visual_fsrvalue = self.robot.get_tactile()
		tactile_values = torch.from_numpy(tactile_values.astype(np.float32)).cuda()
		
		#get finger tip position
		tip_pos = self.robot.get_tip_pos(_action_hora2allegro(INIT_QPOS))
		tip_pos = torch.from_numpy(tip_pos.astype(np.float32)).cuda()

		#get observarion - qpos
		obses = self.robot.get_observation()[:16]
		obses = _obs_allegro2hora(obses)
		obses = torch.from_numpy(obses.astype(np.float32)).cuda()
		if self.sam:
			_, cur_obj_ends = self.robot.get_objects_ends()
			cur_obj_ends = torch.from_numpy(cur_obj_ends).to(self.device)
			cur_obj_ends = cur_obj_ends.reshape([1,1,6])
			obj_ends_history[:,:,:] = cur_obj_ends
				
		proprio_hist_buf = obs_buf_lag_history[:, -self.proprio_len-3:-3, :32].clone()
		tactil_hist_buf = obs_buf_lag_history[:, :, 32:64].clone()
		

		for i in range(timesteps):
			t0 = time.time()
				
			input_dict = {
				'obs': self.running_mean_std(obs_buf.clone().view(1,-1)),
				'proprio_hist': proprio_hist_buf,
				'fingertip_pose': tip_pos[None],
				'obj_ends': obj_ends_history
			}
			obs_buf = obs_buf_lag_history[:, -3:, :32].clone()

			action = self.model.act_inference(input_dict)[0][0]
			action = torch.clamp(action, -1, 1)

			target = prev_target + self.action_scale * action
			target = torch.clamp(target, self.dof_lower, self.dof_upper)
			prev_target = target.clone()
			target = target.type(torch.float32)
			# interact with the hardware
			commands = target.cpu().numpy()
			commands = _action_hora2allegro(commands)
			# allegro.command_joint_position(commands)
			self.robot.set_action(commands)
			# get o_{t+1}
			obses = self.robot.get_observation()[:16]

			#get finger tip position
			tip_pos = self.robot.get_tip_pos(obses)
			tip_pos = torch.from_numpy(tip_pos.astype(np.float32)).cuda()
			
			obses = _obs_allegro2hora(obses)
			print(obses)
			obses = torch.from_numpy(obses.astype(np.float32)).cuda()

			# Get the current tactile values
			tactile_values, visual_fsrvalue = self.robot.get_tactile()
			tactile_values = torch.from_numpy(tactile_values.astype(np.float32)).cuda()


			if self.enable_obj_ends and self.sam:
				prev_obj_ends = obj_ends_history[:, 1:].clone()
				pen_pc, cur_obj_ends = self.robot.get_objects_ends()
				cur_obj_ends = torch.from_numpy(cur_obj_ends).to(self.device)
				cur_obj_ends = cur_obj_ends.reshape([1,1,6])
				obj_ends_history[:] = torch.cat([prev_obj_ends, cur_obj_ends], dim=1)

				if self.visualize:
					pc_img = self.robot.get_point_cloud_with_Tracking_SAM()
					new_points = pc_img[:,:3]
					new_points = o3d.utility.Vector3dVector(new_points)
					end_points = o3d.utility.Vector3dVector(cur_obj_ends)
					pc_img_aug, tactile_pc = self.robot.update_imagnation()
					aug_points = pc_img_aug[:,:3]
					aug_points = o3d.utility.Vector3dVector(aug_points)
					self.pc_visualization(pen_pc=new_points,obj_end_pc=end_points,aug_pc=aug_points)
			prev_obs_buf = obs_buf_lag_history[:, 1:].clone()
			cur_obs_buf = torch.cat([obses[None], target[None], tactile_values[None], tip_pos[None]], dim=-1)
			obs_buf_lag_history = torch.cat([prev_obs_buf, cur_obs_buf[None]], dim=1)

			proprio_hist_buf = obs_buf_lag_history[:,-self.proprio_len-3:-3,:32].clone()
			tactil_hist_buf = obs_buf_lag_history[:, :, 32:64].clone()
		
			dt = time.time() - t0
			count += 1
			print(count)
			print("total_fps",1/dt)
	
	def load(self, fn):
		checkpoint = torch_ext.load_checkpoint(fn)
		self.model.load_state_dict(checkpoint['model'])
		self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
		self.running_mean_std.eval()


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' %(method.__name__, (te - ts) * 1000))
        return result
    return timed
	
if __name__ == '__main__':

	rospy.init_node("hand_grasp")
	network_checkpoint_path = "./checkpoints/vision_student.pth"
	policy = NNRelativeMLPControllerMP(dof_lower=DOF_LOWER_LIMITS, dof_upper=DOF_UPPER_LIMITS, num_actors=1, scale=0.2,
									   proprio_mode=True, enable_obj_ends=True, sam=True, input_mode='proprio-ends')
	# ['proprio-ends', 'proprio-ends-fingertip', 'proprio-tactile', 'proprio-tactile-ends-fingertip']
	policy.load(network_checkpoint_path)
	arm_initial=[210, 70, -100, 180, 150, -30]
	
	while True:    
		policy.robot.xarm_init_set()
		policy.set_initial_target(INIT_QPOS)
		policy.reset()
		policy.set_command(2)
		print("init complete")
		time.sleep(3)

		real_data = policy.deploy(700)
		
		# time.sleep(1 / 30)
		'''
			Action description: The network outputs a [-1, 1]^action_dim vector. 
			
			For relative control (This version), the update is: qpos_target = qpos + action * scale.
			For absolute control, the update is: qpos_target = rescale(action). The action is internally EMAed.	
		'''
		# print(action)
		# sudo chmod -R 777 /dev/ttyUSB0
