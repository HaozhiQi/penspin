import sys
# sys.path.append("")
import torch
import numpy as np

import rospy
import time
import numpy as np

import matplotlib.pyplot as plt

from robot_controller.real_robot.robot import *
from act_agent import ACTPolicy


class Dexpen_ACT:
	def __init__(self, dof_lower, dof_upper,  device='cuda', sam=True, enable_obj_ends=False, input_mode='proprio-ends'):

		self.action_scale = 0.04167
		self.actions_num = 16
		self.control_freq = 18
		self.device = device
		self.sam = sam 
		self.visualize = False
		self.enable_obj_ends = enable_obj_ends
		self.input_mode = input_mode
		self.robot = RealRobot(self.sam)
		
		policy_config = {'lr': 1e-5,
                         'weight_decay': 1e-2,
                         'num_queries': 30,
                         'kl_weight': 200,
                         'hidden_dim': 64,
                         'dim_feedforward': 3200,
                         'lr_backbone': 1e-5,
                         'backbone': 'resnet18',
                         'enc_layers': 4,
                         'dec_layers': 7,
                         'nheads': 8
                         }
		
		self.policy = ACTPolicy(policy_config)
		self.policy.cuda()
     
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
	
	def load(self, weight_path):
		act_network_checkpoint = torch.load(weight_path)
		self.policy.load_state_dict(
            act_network_checkpoint['act_network_state_dict'])
		args = act_network_checkpoint['args']
		return args
		
	def reset(self):
		self.need_init = 1
		# self.reset_rnn()

	def set_initial_target(self, target):
		# target: np.array, absolute qpos.
		self.initial_target = torch.from_numpy(target).float().to(self.device)

	def scale(self, actions):
		return (0.5 * (actions + 1.0) * (self.dof_upper - self.dof_lower) + self.dof_lower)

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

	def deploy(self, timesteps):
		'''
			:param observation: np.array, size = (num_actors, 84, ) dtype=np.float32
			:param deterministic: boolean.
			:return: action: np.array, size = (num_actors, 22, )
		'''
		action_dim = 16
		chunksize = 30
		all_time_actions = torch.zeros([timesteps, timesteps + chunksize, action_dim]).cuda()
		#annotate_init_frame
		# if self.sam:
		# 	print("SAM is ON")
		# 	self.robot.get_point_cloud_with_Tracking_SAM()
		
		for i in range(timesteps):
			t0 = time.time()
	
			qpos = self.robot.get_observation()[:16]
			cur_pos = qpos
			# _, obs = self.robot.get_objects_ends()
			obs = np.zeros([2,3])
			qpos = qpos[None, ...]
			obs = obs[None, ...]
			with torch.no_grad():
				obs = torch.from_numpy(obs).float().to(self.device)
				qpos = torch.from_numpy(qpos).float().to(self.device)
				pred_action = self.policy(obs,qpos)
			
			all_time_actions[[i], i:i + chunksize] = pred_action
			actions_for_current_step = all_time_actions[:, i]
			actions_populated = torch.all(actions_for_current_step != 0, axis=1)
			actions_for_current_step = actions_for_current_step[actions_populated]
			k = 0.01
			exp_weights = np.exp(-k * np.arange(len(actions_for_current_step)))
			exp_weights = exp_weights / exp_weights.sum()
			exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
			raw_action  = (actions_for_current_step * exp_weights).sum(dim=0, keepdim=True)
			raw_action  =  raw_action.cpu().detach().numpy()
			final_acton = cur_pos + raw_action[0]
			self.robot.set_action(final_acton)
			t1 = time.time()
			print("Time taken for one step: ", t1 - t0)

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
	model_path = "/media/binghao/DATA/new_rl_pipeline/rl_pipeline/robot_controller/act_dexpen/checkpoints/epoch_3000.pt"

	act_agent = Dexpen_ACT(dof_lower=DOF_LOWER_LIMITS, dof_upper=DOF_UPPER_LIMITS, enable_obj_ends=True, input_mode='proprio')
	act_agent.load(model_path)
	
	# DexPen position
	arm_initial=[210, 70, -100, 180, 150, -30]
	while True:
		act_agent.set_initial_target(INIT_QPOS)
		act_agent.robot.xarm_init_set()
		act_agent.reset()
		act_agent.set_command(2)
		print("init complete")
		time.sleep(3)
		print("init complete")
		
		act_agent.deploy(200)