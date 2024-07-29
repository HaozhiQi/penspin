from rl_games.common.player import BasePlayer
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.tr_helpers import unsqueeze_obs
import gym
import torch 
from torch import nn
import numpy as np
import pickle
import pickle_utils

def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action

def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


class OpenloopPlayerContinuous(BasePlayer):
    def __init__(self, params):
        BasePlayer.__init__(self, params)
        self.i = 0
        openloop_trajectories = load_data(params['openloop_seq'])
        self.actions = openloop_trajectories[0]['action'][:, :22]
        self.is_rnn = False
        self.mask = [False]
        self.actions_num = self.action_space.shape[0]
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)
        print("Openloop Action", len(self.actions))

    def get_action(self, obs, is_deterministic = False):
        action = self.actions[self.i]
        self.i = (self.i + 1) % len(self.actions)
        return torch.from_numpy(action).float().reshape(1, -1).repeat(obs.size(0), 1)
    def restore(self, fn):
        return
    def reset(self):
        self.i = 0
        return

class PpoPlayerContinuous(BasePlayer):
    def __init__(self, params):
        BasePlayer.__init__(self, params)
        self.network = self.config['network']
        self.actions_num = self.action_space.shape[0] 
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)

        obs_shape = self.obs_shape
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        } 
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

    def get_action(self, obs, is_deterministic = False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            'rnn_states' : self.states
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_deterministic:
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        print("Loading checkpoint")
        self.model.load_state_dict(checkpoint['model'])
        print("check", checkpoint['model'].keys())
        if self.normalize_input: #and 'running_mean_std' in checkpoint:
            print("Loaded")
            if 'running_mean_std.running_mean_std.obs.running_mean' in checkpoint['model']:
                self.model.running_mean_std.running_mean_std.obs.count.data = (
                    checkpoint['model']['running_mean_std.running_mean_std.obs.count'].data).to(self.device)
                self.model.running_mean_std.running_mean_std.obs.running_mean.data = (
                    checkpoint['model']['running_mean_std.running_mean_std.obs.running_mean'].data).to(self.device)
                self.model.running_mean_std.running_mean_std.obs.running_var.data = (
                    checkpoint['model']['running_mean_std.running_mean_std.obs.running_var'].data).to(self.device)
                self.model.running_mean_std.running_mean_std.pointcloud.count.data = (
                    checkpoint['model']['running_mean_std.running_mean_std.pointcloud.count'].data).to(self.device)
                self.model.running_mean_std.running_mean_std.pointcloud.running_mean.data = (
                    checkpoint['model']['running_mean_std.running_mean_std.pointcloud.running_mean'].data).to(self.device)
                self.model.running_mean_std.running_mean_std.pointcloud.running_var.data = (
                    checkpoint['model']['running_mean_std.running_mean_std.pointcloud.running_var'].data).to(self.device)
                # self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
            else:
                self.model.running_mean_std.count.data = (
                    checkpoint['model']['running_mean_std.count'].data).to(self.device)
                self.model.running_mean_std.running_mean.data = (
                    checkpoint['model']['running_mean_std.running_mean'].data).to(self.device)
                self.model.running_mean_std.running_var.data = (
                    checkpoint['model']['running_mean_std.running_var'].data).to(self.device)
    def reset(self):
        self.init_rnn()

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_deterministic = self.is_deterministic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True
            #print('setting agent weights for selfplay')
            # self.env.create_agent(self.env.config)
            # self.env.set_weights(range(8),self.get_weights())

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn

        is_openloop_trajectory_enough = False
        openloop_trajectory_saved = False

        env_dones = np.ones(16) * 2
        env_reward = np.zeros(16)
        end_flag = False

        n_object_dict = {'C': 16, 27: 27, 'poly': 162, 'non-convex': 40, 'cuboid': 1, 'cross': 5, 'ball': 1, 'block': 1, 'asymmetry': 172, 'duck': 1, 'giraffe': 1, 'symmetry': 1, 'designer-cube': 1, "cross_bmr": 1, "cross5": 1, "designer-cube2": 1, "cross_y": 1, "cross3": 1, "cross_t": 1, "designer-cube3": 1,
                            'i': 1, 'c': 1, 'r': 1, 'a': 1, 'mouse': 1}
        n_object = n_object_dict[self.env.cfg['env']['objSet']]
        object_count_dict = {'game_count': np.zeros(n_object), 'reward_sum': np.zeros(n_object), 'step_sum': np.zeros(n_object)}
        n_games = 500
        print(n_games, self.max_steps, self.n_game_life)

        for game_idx in range(n_games):
            # if all(object_count_dict['game_count'] >= 1):
            #     break
            if games_played >= n_games:
                break

            obses = self.env_reset(self.env)
            batch_size = 1
            batch_size = self.get_batch_size(obses, batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False

            open_loop_trajectories_action = []
            open_loop_trajectories_done = []
            open_loop_trajectories_state = []

            for n in range(self.max_steps):
                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(
                        obses, masks, is_deterministic)
                else:
                    action = self.get_action(obses, is_deterministic)

                #if n <= 10:
                #    print(n, action)
                #    print(obses)
                #else:
                #    exit(0)
                next_obses, r, done, info = self.env_step(self.env, action)
                cr += r
                steps += 1

                # Record some open loop trajectories.
                if not is_openloop_trajectory_enough:
                    open_loop_trajectories_action.append(action) # [num_envs, action_dim]
                    open_loop_trajectories_done.append(done)
                    open_loop_trajectories_state.append(obses)

                if len(open_loop_trajectories_action) > 400:
                    is_openloop_trajectory_enough = True

                # if is_openloop_trajectory_enough:
                #     if not openloop_trajectory_saved:
                #         open_loop_trajectories_action = torch.stack(open_loop_trajectories_action, dim=0)
                #         open_loop_trajectories_done = torch.stack(open_loop_trajectories_done, dim=0)
                #         open_loop_trajectories_state = torch.stack(open_loop_trajectories_state, dim=0)

                        # pickle_utils.save_data({'obs': open_loop_trajectories_state.detach().cpu().numpy(),
                        #                         'act': open_loop_trajectories_action.detach().cpu().numpy(),
                        #                         'done': open_loop_trajectories_done.detach().cpu().numpy()},
                        #                         self.action_savepath)

                        #print("Data saved.")

                    openloop_trajectory_saved = True

                if render:
                    self.env.render(mode='human')
                    time.sleep(self.render_sleep)

                obses = next_obses
                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                default_mode = True
                if done_count > 0:
                    if default_mode:
                        if self.is_rnn:
                            for s in self.states:
                                s[:, all_done_indices, :] = s[:,all_done_indices, :] * 0.0

                        cur_rewards = cr[done_indices].sum().item()
                        cur_steps = steps[done_indices].sum().item()

                        # Fetch object information
                        obj_idx_info = self.env.get_internal_info('obj')
                        for done_index in done_indices:
                            object_count_dict['game_count'][obj_idx_info[done_index][:, 0].item()] += 1
                            object_count_dict['reward_sum'][obj_idx_info[done_index][:, 0].item()] += cr[done_index].item()
                            object_count_dict['step_sum'][obj_idx_info[done_index][:, 0].item()] += steps[done_index].item()
                        # print(self.env.get_internal_info('obj'))

                        cr = cr * (1.0 - done.float())
                        steps = steps * (1.0 - done.float())
                        sum_rewards += cur_rewards
                        sum_steps += cur_steps

                        game_res = 0.0
                        if isinstance(info, dict):
                            if 'battle_won' in info:
                                print_game_res = True
                                game_res = info.get('battle_won', 0.5)
                            if 'scores' in info:
                                print_game_res = True
                                game_res = info.get('scores', 0.5)

                        if self.print_stats:
                            if print_game_res:
                                print('reward:', cur_rewards/done_count,
                                      'steps:', cur_steps/done_count, 'w:', game_res)
                            else:
                                print('reward:', cur_rewards/done_count,
                                      'steps:', cur_steps/done_count)

                        sum_game_res += game_res
                        print(games_played, object_count_dict['game_count'])
                        if batch_size//self.num_agents == 1 or games_played >= n_games:
                            break
                    else:
                        if self.is_rnn:
                            for s in self.states:
                                s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0

                        #break_flag = True
                        #print(env_dones, list(done_indices))
                        for d in list(done_indices):
                            if env_dones[d] <= 0:
                                continue
                            else:
                                env_reward[d] += cr[d].item()
                                env_dones[d] -= 1
                                #break_flag = False

                        if len(np.where(env_dones > 0)[0]) < 0:
                            break_flag = True
                        else:
                            break_flag = False

                        if break_flag:
                            end_flag = True
                            break

                        cur_rewards = cr[done_indices].sum().item()
                        cur_steps = steps[done_indices].sum().item()

                        cr = cr * (1.0 - done.float())
                        steps = steps * (1.0 - done.float())
                        sum_rewards += cur_rewards
                        sum_steps += cur_steps

                        game_res = 0.0
                        if self.print_stats:
                            if print_game_res:
                                print(cur_steps)
                                print('reward:', cur_rewards / done_count,
                                      'steps:', cur_steps / done_count, 'w:', game_res)
                            else:
                                print('reward:', cur_rewards / done_count,
                                      'steps:', cur_steps / done_count)

                        sum_game_res += game_res
                        if batch_size // self.num_agents == 1 or games_played >= n_games:
                            break
            if end_flag:
                break

        mask = np.where(object_count_dict['game_count']>0, 1, 0)
        av_rew = np.where(object_count_dict['game_count']>0, object_count_dict['reward_sum']/object_count_dict['game_count'], 0)
        av_step = np.where(object_count_dict['game_count']>0, object_count_dict['step_sum']/object_count_dict['game_count'], 0)
        print("av of av rewards:", np.average(av_rew, weights=mask),
            "av of av steps:", np.average(av_step, weights=mask))
        print("ALL", env_reward.sum() / 32)
        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps /
                  games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life,
                  'av steps:', sum_steps / games_played * n_game_life)

class PpoPlayerContinuousCollect(BasePlayer):
    def __init__(self, params):
        BasePlayer.__init__(self, params)
        self.network = self.config['network']
        self.actions_num = self.action_space.shape[0]
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)

        obs_shape = self.obs_shape
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

    def get_action(self, obs, is_deterministic = False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs,
            'rnn_states': self.states
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_deterministic:
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def reset(self):
        self.init_rnn()
        self.collect_trajectories_action = None
        self.collect_trajectories_done = None
        self.collect_trajectories_state = None
        if self.is_rnn:
            self.collect_trajectories_rnn = None
        self.collect_trajectories_gt = None

        self.collect_trajectories_qpos = None
        self.collect_trajectories_target = None
        self.collect_trajectories_contact = None
        self.collect_trajectories_obj_class = None

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_deterministic = self.is_deterministic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True
            # print('setting agent weights for selfplay')
            # self.env.create_agent(self.env.config)
            # self.env.set_weights(range(8),self.get_weights())

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn

        collect_trajectories_action = []
        collect_trajectories_done = []
        collect_trajectories_state = []
        if self.is_rnn:
            collect_trajectories_rnn = []
        collect_trajectories_gt = []
        collect_trajectories_target = []
        collect_trajectories_qpos = []
        collect_trajectories_contact = []
        collect_trajectories_obj_class = []
        collect_trajectories_init_quat = []

        for game_idx in range(n_games):
            if games_played >= n_games:
                break

            obses = self.env_reset(self.env)
            batch_size = 1
            batch_size = self.get_batch_size(obses, batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False

            for n in range(self.max_steps):
                # print(len(self.states), self.states[0].shape, self.states[1].shape)
                if self.is_rnn:
                    current_rnn_state = torch.cat(self.states, dim=0)

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(
                        obses, masks, is_deterministic)
                else:
                    action = self.get_action(obses, is_deterministic)

                groundtruth = self.get_env_internal_state(self.env)
                next_obses, r, done, info = self.env_step(self.env, action)

                qpos = self.get_env_internal_info(self.env, 'qpos')
                # print(qpos)
                target = self.get_env_internal_info(self.env, 'target')
                contact = self.get_env_internal_info(self.env, 'contact')
                obj_class = self.get_env_internal_info(self.env, 'obj')
                init_quat = self.get_env_internal_info(self.env, 'qinit')
                # This will record the execution trajectory....

                cr += r
                steps += 1

                # Record some open loop trajectories.
                collect_trajectories_action.append(action.detach().cpu().numpy())  # [num_envs, action_dim]
                collect_trajectories_done.append(done.detach().cpu().numpy())
                if self.is_rnn:
                    collect_trajectories_rnn.append(current_rnn_state.detach().cpu().numpy())
                collect_trajectories_state.append(obses.detach().cpu().numpy())
                collect_trajectories_gt.append(groundtruth.detach().cpu().numpy())
                collect_trajectories_qpos.append(qpos.detach().cpu().numpy())
                collect_trajectories_target.append(target.detach().cpu().numpy())
                collect_trajectories_contact.append(contact.detach().cpu().numpy())
                collect_trajectories_obj_class.append(obj_class.detach().cpu().numpy())
                collect_trajectories_init_quat.append(init_quat.detach().cpu().numpy())
                #print(obj_class)
                if render:
                    self.env.render(mode='human')
                    time.sleep(self.render_sleep)

                obses = next_obses
                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if self.print_stats:
                        if print_game_res:
                            print(cur_steps)
                            print('reward:', cur_rewards / done_count,
                                  'steps:', cur_steps / done_count, 'w:', game_res)
                        else:
                            print('reward:', cur_rewards / done_count,
                                  'steps:', cur_steps / done_count)

                    sum_game_res += game_res
                    if batch_size // self.num_agents == 1 or games_played >= n_games:
                        break

        # We have collect enough trajectories.
        # Save them
        # self.collect_trajectories_action = torch.stack(collect_trajectories_action, dim=0)
        # self.collect_trajectories_done = torch.stack(collect_trajectories_done, dim=0)
        # self.collect_trajectories_state = torch.stack(collect_trajectories_state, dim=0)
        # if self.is_rnn:
        #     self.collect_trajectories_rnn = torch.stack(collect_trajectories_rnn, dim=0)
        # self.collect_trajectories_gt = torch.stack(collect_trajectories_gt, dim=0)
        #
        # self.collect_trajectories_qpos = torch.stack(collect_trajectories_qpos, dim=0)
        # self.collect_trajectories_target = torch.stack(collect_trajectories_target, dim=0)
        # self.collect_trajectories_contact = torch.stack(collect_trajectories_contact, dim=0)
        # self.collect_trajectories_obj_class = torch.stack(collect_trajectories_obj_class, dim=0)

        self.collect_trajectories_action = np.stack(collect_trajectories_action, axis=0)
        self.collect_trajectories_done = np.stack(collect_trajectories_done, axis=0)
        self.collect_trajectories_state = np.stack(collect_trajectories_state, axis=0)
        if self.is_rnn:
            self.collect_trajectories_rnn = np.stack(collect_trajectories_rnn, axis=0)
        self.collect_trajectories_gt = np.stack(collect_trajectories_gt, axis=0)

        self.collect_trajectories_qpos = np.stack(collect_trajectories_qpos, axis=0)
        self.collect_trajectories_target = np.stack(collect_trajectories_target, axis=0)
        self.collect_trajectories_contact = np.stack(collect_trajectories_contact, axis=0)
        self.collect_trajectories_obj_class = np.stack(collect_trajectories_obj_class, axis=0)
        self.collect_trajectories_init_quat = np.stack(collect_trajectories_init_quat, axis=0)


        # Maybe we should not save this.
        # pickle_utils.save_data({'obs': collect_trajectories_state.detach().cpu().numpy(),
        #                         'act': collect_trajectories_action.detach().cpu().numpy(),
        #                         'done': collect_trajectories_done.detach().cpu().numpy(),
        #                         'rnn': collect_trajectories_rnn.detach().cpu().numpy(),
        #                         'gt': collect_trajectories_gt.detach().cpu().numpy()},
        #                        self.action_savepath)
        print(self.collect_trajectories_state.shape, self.collect_trajectories_action.shape,
              self.collect_trajectories_gt.shape, self.collect_trajectories_done.shape,
              self.collect_trajectories_qpos.shape,
              self.collect_trajectories_target.shape)

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps /
                  games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life,
                  'av steps:', sum_steps / games_played * n_game_life)

    def post_run(self, runner):
        print("Player post-running procedure, saving record data.")
        if self.is_rnn:
            runner.set_record_data(state=self.collect_trajectories_state,
                                   action=self.collect_trajectories_action,
                                   gt=self.collect_trajectories_gt,
                                   done=self.collect_trajectories_done,
                                   rnn=self.collect_trajectories_rnn,
                                   qpos=self.collect_trajectories_qpos,
                                   target=self.collect_trajectories_target,
                                   contact=self.collect_trajectories_contact,
                                   qinit=self.collect_trajectories_init_quat)
        else:
            runner.set_record_data(state=self.collect_trajectories_state,
                                   action=self.collect_trajectories_action,
                                   gt=self.collect_trajectories_gt,
                                   done=self.collect_trajectories_done,
                                   qpos=self.collect_trajectories_qpos,
                                   target=self.collect_trajectories_target,
                                   contact=self.collect_trajectories_contact,
                                   obj=self.collect_trajectories_obj_class,
                                   qinit=self.collect_trajectories_init_quat)
        return

class PpoPlayerDiscrete(BasePlayer):
    def __init__(self, params):
        BasePlayer.__init__(self, params)

        self.network = self.config['network']
        if type(self.action_space) is gym.spaces.Discrete:
            self.actions_num = self.action_space.n
            self.is_multi_discrete = False
        if type(self.action_space) is gym.spaces.Tuple:
            self.actions_num = [action.n for action in self.action_space]
            self.is_multi_discrete = True
        self.mask = [False]
        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)
        obs_shape = self.obs_shape
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }

        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

    def get_masked_action(self, obs, action_masks, is_deterministic = True):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        action_masks = torch.Tensor(action_masks).to(self.device).bool()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            'action_masks' : action_masks,
            'rnn_states' : self.states
        }
        self.model.eval()

        with torch.no_grad():
            res_dict = self.model(input_dict)
        logits = res_dict['logits']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if self.is_multi_discrete:
            if is_deterministic:
                action = [torch.argmax(logit.detach(), axis=-1).squeeze() for logit in logits]
                return torch.stack(action,dim=-1)
            else:    
                return action.squeeze().detach()
        else:
            if is_deterministic:
                return torch.argmax(logits.detach(), axis=-1).squeeze()
            else:    
                return action.squeeze().detach()

    def get_action(self, obs, is_deterministic = False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)

        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            'rnn_states' : self.states
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        logits = res_dict['logits']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if self.is_multi_discrete:
            if is_deterministic:
                action = [torch.argmax(logit.detach(), axis=1).squeeze() for logit in logits]
                return torch.stack(action,dim=-1)
            else:    
                return action.squeeze().detach()
        else:
            if is_deterministic:
                return torch.argmax(logits.detach(), axis=-1).squeeze()
            else:    
                return action.squeeze().detach()

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def reset(self):
        self.init_rnn()


class SACPlayer(BasePlayer):
    def __init__(self, params):
        BasePlayer.__init__(self, params)
        self.network = self.config['network']
        self.actions_num = self.action_space.shape[0] 
        self.action_range = [
            float(self.env_info['action_space'].low.min()),
            float(self.env_info['action_space'].high.max())
        ]

        obs_shape = self.obs_shape
        self.normalize_input = False
        config = {
            'obs_dim': self.env_info["observation_space"].shape[0],
            'action_dim': self.env_info["action_space"].shape[0],
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': False,
            'normalize_input': self.normalize_input,
        }  
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.sac_network.actor.load_state_dict(checkpoint['actor'])
        self.model.sac_network.critic.load_state_dict(checkpoint['critic'])
        self.model.sac_network.critic_target.load_state_dict(checkpoint['critic_target'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def get_action(self, obs, is_deterministic=False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        dist = self.model.actor(obs)
        actions = dist.sample() if is_deterministic else dist.mean
        actions = actions.clamp(*self.action_range).to(self.device)
        if self.has_batch_dimension == False:
            actions = torch.squeeze(actions.detach())
        return actions

    def reset(self):
        pass
