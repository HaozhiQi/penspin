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

class EpisodicDataset(Dataset):
    def __init__(self, episode_ids, data_path=None):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.path = data_path
        with h5py.File(data_path, 'r') as root:
            self.demo_data = root[f"episode_{self.episode_ids[0]}"]
            self.chunk_size = len(self.demo_data['obs'])
            self.dummy_data = {'obs': self.demo_data['obs'][0],'robot_qpos': self.demo_data['robot_qpos'][0], 'action': self.demo_data['action'][0]}
    
    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode for now
        episode_id = self.episode_ids[index]
    
        data_path = self.path
        with h5py.File(data_path, 'r') as root:
            self.episode_data  = root[f"episode_{episode_id}"]  
            self.obs = self.episode_data['obs']
            self.robot_qpos = self.episode_data['robot_qpos']
            self.action = self.episode_data['action']

            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(self.chunk_size)

            # get observation at start_ts only
            obs = self.obs[start_ts]
            robot_qpos = self.robot_qpos[start_ts]
            
            # get all actions after and including start_ts
            action_len = self.chunk_size - start_ts
            action = self.action[start_ts:]
            padded_action =np.zeros((self.chunk_size,len(action[0])), dtype=np.float32)
            padded_action[:action_len] = action
            is_pad = np.zeros(self.chunk_size)
            is_pad[action_len:] = 1

        obs = torch.from_numpy(obs).float()
        robot_qpos = torch.from_numpy(robot_qpos).float()
        padded_action = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
            
        return obs, robot_qpos, padded_action, is_pad


def prepare_real_pen_data(real_dataset_folder=None, real_batch_size=None, val_ratio = 0.1, seed = 0, chunk_size = 30):
    
    real_demo_file = os.path.join(real_dataset_folder, "real_data.h5")
    real_demo_processed_file = os.path.join(real_dataset_folder, "real_data_processed.h5")
    total_episodes = 0
    print('=== chunk the data into episodes ===')
    with h5py.File(real_demo_file, 'r') as root:
        with h5py.File(real_demo_processed_file, 'w') as root_write:
            for demo_idx in range(len(root.keys())):
                demo_data = root[f"replay_demon_{demo_idx}"]
                total_episodes_per_demo = len(demo_data['qpos']) // chunk_size
                for i in range(total_episodes_per_demo):
                    episode_data = root_write.create_group(f"episode_{i+total_episodes}")
                    # now using the obj_ends as observation
                    episode_data.create_dataset('obs', data=demo_data['obj_ends'][i*chunk_size:(i+1)*chunk_size])
                    episode_data.create_dataset('robot_qpos', data=demo_data['qpos'][i*chunk_size:(i+1)*chunk_size])
                    episode_data.create_dataset('action', data=demo_data['action'][i*chunk_size:(i+1)*chunk_size])
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
        
    bc_train_set = EpisodicDataset(episode_ids=train_demo_idx, data_path=data_path)
    bc_validation_set = EpisodicDataset(episode_ids=validation_demo_idx, data_path=data_path)
   
    bc_train_dataloader = DataLoader(bc_train_set, batch_size=batch_size, shuffle=True)
    val_batch_size = batch_size if batch_size < len(bc_validation_set) else len(bc_validation_set)
    bc_validation_dataloader = DataLoader(bc_validation_set, batch_size=val_batch_size, shuffle=False)
    
    it_per_epoch = len(bc_train_set) // batch_size
    print('  ', 'total number of training samples', len(bc_train_set))
    print('  ', 'total number of validation samples', len(bc_validation_set))
    print('  ', 'number of iters per epoch', it_per_epoch)
    return it_per_epoch, bc_train_set, bc_train_dataloader, bc_validation_dataloader
