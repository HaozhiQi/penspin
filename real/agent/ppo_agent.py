# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://[arxiv link]
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: RLGames
# Copyright (c) 2019 Denys88
# Licence under MIT License
# https://github.com/Denys88/rl_games/
# --------------------------------------------------------

import os
import time
import torch
import torch.distributed as dist
import numpy as np

import sys
current_path = os.path.dirname(__file__)
if current_path=='':
    sys.path.append(sys.path[0]+'/../..')
else:
    sys.path.append(current_path+'/../..')

from penspin.algo.models.models import ActorCritic
from penspin.algo.models.running_mean_std import RunningMeanStd

def _action_hora2allegro(actions):
    cmd_act = actions.clone()
    cmd_act[[4, 5, 6, 7]] = actions[[8, 9, 10, 11]]
    cmd_act[[12, 13, 14, 15]] = actions[[4, 5, 6, 7]]
    cmd_act[[8, 9, 10, 11]] = actions[[12, 13, 14, 15]]
    return cmd_act

class FinetunePPO(object):
    def __init__(self, args):
        self.obs_shape = (96,)
        # ---- Model ----
        net_config = {
            'actor_units': [512, 256, 128],
            'priv_mlp_units': [256, 128, 8],
            'actions_num': 16,
            'input_shape': (96,),
            'priv_info': False,
            'proprio_adapt': False,
            'priv_info_dim': 61,
            'critic_info_dim': 100,
            'asymm_actor_critic': False,
            'point_mlp_units': [32, 32, 32],
            'use_fine_contact': False,
            'contact_mlp_units': [32, 32, 32],
            'use_point_transformer': False,
            'multi_axis': False,
            'proprio_mode': True,
            'input_mode': 'proprio',
            'proprio_len': 24,
            'student': True,
            'use_point_cloud_info': False,
        }
        self.model = ActorCritic(net_config)
        self.model.cuda()
        self.running_mean_std = RunningMeanStd(self.obs_shape).cuda()
        self.param_dicts = [
        { "params": [p for n, p in self.model.named_parameters() if "actor_mlp" in n and p.requires_grad]}     
        ]
        self.lr = args['lr']
        self.weight_decay = args['weight_decay']
        #self.optimizer = torch.optim.Adam(self.param_dicts, self.lr, weight_decay=self.weight_decay)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=self.weight_decay)

    def set_eval(self):
        self.model.eval()
        self.running_mean_std.eval()

    def set_train(self):
        self.model.train()
        self.running_mean_std.train()

    def model_act(self, obs, proprio_hist, finger_tip_pos):
        processed_obs = self.running_mean_std(obs)
        proprio_hist = proprio_hist
        priv_info = torch.zeros((obs.shape[0], 61)).cuda()
        priv_info[..., 16:28] = finger_tip_pos
        input_dict = {
            'priv_info': priv_info,
            'obs': processed_obs,
            'proprio_hist': proprio_hist,
        }
        res_dict = self.model.act(input_dict)
        actions = torch.clamp(res_dict['mus'], -1.0, 1.0)

        return _action_hora2allegro(actions)
   
    def compute_loss(self, obs, proprio_hist, finger_tip_pos, action):
        actions_hat = self.model_act(obs, proprio_hist, finger_tip_pos).requires_grad_()
        loss = torch.nn.functional.l1_loss(actions_hat, action).mean().cuda()
        return loss
    
    def update_policy(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def load(self, path):
        print("loading demonstration checkpoint from path", path)
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        self.running_mean_std.train()

    def save(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'running_mean_std': self.running_mean_std.state_dict(),
        }, path)
    
    def evaluate(self, obs, proprio_hist, finger_tip_pos, action=None):
        actions_hat = self.model_act(obs, proprio_hist, finger_tip_pos)

        if action is not None:
            all_l1 = torch.nn.functional.l1_loss(actions_hat, action, reduction='none').mean()
          
            return all_l1.detach().cpu().item()

        return actions_hat.detach().cpu().numpy()