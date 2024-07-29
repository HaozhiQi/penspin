import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
import h5py
import pickle
import random
import shutil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def _obs_allegro2hora(obses):
    obs_index = obses[0:4]
    obs_middle = obses[4:8]
    obs_ring = obses[8:12]
    obs_thumb = obses[12:16]
    obses = np.concatenate([obs_index, obs_thumb, obs_middle, obs_ring]).astype(np.float32)
    return obses

class PPODataset(Dataset):
    def __init__(self, episode_ids, data_path=None):
        super(PPODataset).__init__()
        self.episode_ids = episode_ids
        self.path = data_path
        with h5py.File(data_path, 'r') as root:
            self.demo_data = root[f"episode_{self.episode_ids[0]}"]
            self.chunk_size = len(self.demo_data['obs'])
            self.dummy_data = {'obs': self.demo_data['obs'][0], 'proprio_hist':self.demo_data['proprio_hist'][0],\
                               'finger_tip_pos': self.demo_data['finger_tip_pos'][0], 'action': self.demo_data['action'][0]}
                               
    
    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        data_path = self.path
        with h5py.File(data_path, 'r') as root:
            self.episode_data  = root[f"episode_{episode_id}"]  
            obs = self.episode_data['obs'][()] 
            proprio_hist = self.episode_data['proprio_hist'][()] 
            finger_tip_pos = self.episode_data['finger_tip_pos'][()] 
            action = self.episode_data['action'][()] 

        obs = torch.from_numpy(obs).float()
        proprio_hist = torch.from_numpy(proprio_hist).float()
        finger_tip_pos = torch.from_numpy(finger_tip_pos).float()
        action = torch.from_numpy(action).float()
        
            
        return obs, proprio_hist, finger_tip_pos, action


def prepare_real_pen_data_ppo(real_dataset_folder=None, real_batch_size=None, val_ratio = 0.1, seed = 0):
    
    real_demo_file = os.path.join(real_dataset_folder, "real_data.h5")
    real_demo_processed_file = os.path.join(real_dataset_folder, "real_data_processed.h5")
    total_episodes = 0
    print('=== chunk the data into episodes ===')
    with h5py.File(real_demo_file, 'r') as root:
        with h5py.File(real_demo_processed_file, 'w') as root_write:
            for demo_idx in range(len(root.keys())):
                demo_data = root[f"replay_demon_{demo_idx}"]
                processed_obses = []
                for i in range(len(demo_data['qpos'])):
                    processed_qpos = _obs_allegro2hora(demo_data['qpos'][i])
                    processed_cur_target = _obs_allegro2hora(demo_data['current_target'][i])
                    processed_obses.append(np.concatenate([processed_qpos, processed_cur_target]))                                      
                total_episodes_per_demo = len(demo_data['qpos']) - 30 + 1
                for i in range(total_episodes_per_demo):
                    episode_data = root_write.create_group(f"episode_{i+total_episodes}")
                    # now using the obj_ends as observation
                    episode_data.create_dataset('obs', data=np.concatenate(processed_obses[i:i+3]))
                    episode_data.create_dataset('proprio_hist', data=processed_obses[i:i+30])
                    episode_data.create_dataset('finger_tip_pos', data=demo_data['finger_tip_pose'][i])
                    episode_data.create_dataset('action', data=demo_data['action'][i])
                total_episodes += total_episodes_per_demo
        
    print('  ', 'total number of demos', total_episodes)
    print('=== Loading Real trajectories ===')
    
    it_per_epoch, bc_train_set, bc_train_dataloader, bc_validation_dataloader = prepare_data(real_demo_processed_file, total_episodes, real_batch_size, val_ratio, seed)
            
    Prepared_Data = {"it_per_epoch": it_per_epoch, "bc_train_set": bc_train_set, "bc_train_dataloader": bc_train_dataloader, 
                    "bc_validation_dataloader": bc_validation_dataloader, "total_episodes": total_episodes}

    return Prepared_Data

def prepare_data(data_path, total_episodes, batch_size, val_ratio = 0.1, seed = 0):
    

    set_seed(seed)
    random_demo_id = np.random.permutation(total_episodes)
    train_demo_idx = random_demo_id[:int(len(random_demo_id)*(1-val_ratio))]
    validation_demo_idx = random_demo_id[int(len(random_demo_id)*(1-val_ratio)):]
        
    bc_train_set = PPODataset(episode_ids=train_demo_idx, data_path=data_path)
    bc_validation_set = PPODataset(episode_ids=validation_demo_idx, data_path=data_path)
   
    bc_train_dataloader = DataLoader(bc_train_set, batch_size=batch_size, shuffle=True)
    val_batch_size = batch_size if batch_size < len(bc_validation_set) else len(bc_validation_set)
    bc_validation_dataloader = DataLoader(bc_validation_set, batch_size=val_batch_size, shuffle=False)
    
    it_per_epoch = len(bc_train_set) // batch_size
    print('  ', 'total number of training samples', len(bc_train_set))
    print('  ', 'total number of validation samples', len(bc_validation_set))
    print('  ', 'number of iters per epoch', it_per_epoch)
    return it_per_epoch, bc_train_set, bc_train_dataloader, bc_validation_dataloader
