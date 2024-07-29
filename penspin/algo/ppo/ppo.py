# --------------------------------------------------------
# Lessons from Learning to Spin “Pens”
# Written by Paper Authors
# Copyright (c) 2024 All Authors
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
import numpy as np

from penspin.algo.ppo.experience import ExperienceBuffer
from penspin.algo.models.models import ActorCritic
from penspin.algo.models.running_mean_std import RunningMeanStd

from penspin.utils.misc import AverageScalarMeter

from tensorboardX import SummaryWriter


class PPO(object):
    def __init__(self, env, output_dif, full_config):
        self.device = full_config['rl_device']
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo
        # ---- build environment ----
        self.env = env
        self.num_actors = self.ppo_config['num_actors']
        action_space = self.env.action_space
        self.actions_num = action_space.shape[0]
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.device)
        self.observation_space = self.env.observation_space
        self.obs_shape = self.observation_space.shape
        # ---- Priv Info ----
        self.priv_info_dim = self.env.priv_info_dim
        self.priv_info = self.ppo_config['priv_info']
        self.proprio_adapt = self.ppo_config['proprio_adapt']
        # ---- Critic Info
        self.asymm_actor_critic = self.ppo_config['asymm_actor_critic']
        self.critic_info_dim = self.ppo_config['critic_info_dim']
        # ---- Point Cloud Info
        self.point_cloud_buffer_dim = self.env.point_cloud_buffer_dim
        # proprio-mode
        self.proprio_mode = self.ppo_config['proprio_mode']
        self.input_mode = self.ppo_config['input_mode']
        self.proprio_len = self.ppo_config['proprio_len']
        self.use_point_cloud_info = self.ppo_config['use_point_cloud_info']
        # ---- Output Dir ----
        # allows us to specify a folder where all experiments will reside
        self.output_dir = output_dif
        self.nn_dir = os.path.join(self.output_dir, 'stage1_nn')
        self.tb_dif = os.path.join(self.output_dir, 'stage1_tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dif, exist_ok=True)
        # ---- Model ----
        net_config = {
            'actor_units': self.network_config.mlp.units,
            'priv_mlp_units': self.network_config.priv_mlp.units,
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'priv_info': self.priv_info,
            'proprio_adapt': self.proprio_adapt,
            'priv_info_dim': self.priv_info_dim,
            'critic_info_dim': self.critic_info_dim,
            'asymm_actor_critic': self.asymm_actor_critic,
            'point_cloud_sampled_dim': self.point_cloud_buffer_dim,
            'point_mlp_units': self.network_config.point_mlp.units,
            'use_fine_contact': self.env.contact_input == 'fine',
            'contact_mlp_units': self.network_config.contact_mlp.units,
            'use_point_transformer': self.network_config.use_point_transformer,
            'use_point_cloud_info': self.use_point_cloud_info,
            'proprio_mode': self.proprio_mode,
            'input_mode': self.input_mode,
            'proprio_len': self.proprio_len,
            'student': self.ppo_config.distill,
        }
        self.model = ActorCritic(net_config)
        self.model.to(self.device)

        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.priv_mean_std = RunningMeanStd(self.priv_info_dim).to(self.device)
        self.proprio_dim = 32
        if self.ppo_config.distill:
            self.point_cloud_mean_std = RunningMeanStd(6,).to(self.device)
        else:
            self.point_cloud_mean_std = RunningMeanStd(3,).to(self.device)
        self.value_mean_std = RunningMeanStd((1,)).to(self.device)
        # ---- Optim ----
        self.last_lr = float(self.ppo_config['learning_rate'])
        self.weight_decay = self.ppo_config.get('weight_decay', 0.0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.last_lr, weight_decay=self.weight_decay)
        # ---- PPO Train Param ----
        self.e_clip = self.ppo_config['e_clip']
        self.clip_value = self.ppo_config['clip_value']
        self.entropy_coef = self.ppo_config['entropy_coef']
        self.critic_coef = self.ppo_config['critic_coef']
        self.bounds_loss_coef = self.ppo_config['bounds_loss_coef']
        self.distill_loss_coef = self.ppo_config['distill_loss_coef']
        self.gamma = self.ppo_config['gamma']
        self.tau = self.ppo_config['tau']
        self.truncate_grads = self.ppo_config['truncate_grads']
        self.grad_norm = self.ppo_config['grad_norm']
        self.value_bootstrap = self.ppo_config['value_bootstrap']
        self.normalize_advantage = self.ppo_config['normalize_advantage']
        self.normalize_input = self.ppo_config['normalize_input']
        self.normalize_value = self.ppo_config['normalize_value']
        self.normalize_priv = self.ppo_config['normalize_priv']
        self.normalize_point_cloud = self.ppo_config['normalize_point_cloud']
        # ---- PPO Collect Param ----
        self.horizon_length = self.ppo_config['horizon_length']
        self.batch_size = self.horizon_length * self.num_actors
        self.minibatch_size = self.ppo_config['minibatch_size']
        self.mini_epochs_num = self.ppo_config['mini_epochs']
        assert self.batch_size % self.minibatch_size == 0 or full_config.test
        # ---- scheduler ----
        self.kl_threshold = self.ppo_config['kl_threshold']
        self.scheduler = AdaptiveScheduler(self.kl_threshold)
        # ---- Snapshot
        self.save_freq = self.ppo_config['save_frequency']
        self.save_best_after = self.ppo_config['save_best_after']
        # ---- Tensorboard Logger ----
        self.extra_info = {}
        writer = SummaryWriter(self.tb_dif)
        self.writer = writer

        # ---- Rollout GIFs ----
        self.gif_frame_counter = 0
        self.gif_save_every_n = 7500
        self.gif_save_length = 600
        self.gif_frames = []

        self.episode_rewards = AverageScalarMeter(20000)
        self.episode_lengths = AverageScalarMeter(20000)
        self.obs = None
        self.epoch_num = 0
        self.storage = ExperienceBuffer(
            self.num_actors, self.horizon_length, self.batch_size, self.minibatch_size, self.obs_shape[0],
            self.actions_num, self.priv_info_dim, self.critic_info_dim, self.point_cloud_buffer_dim, self.device,
            self.proprio_dim,self.proprio_len
        )

        batch_size = self.num_actors
        current_rewards_shape = (batch_size, 1)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)
        self.agent_steps = 0
        self.max_agent_steps = self.ppo_config['max_agent_steps']
        self.best_rewards = -10000
        # ---- Timing
        self.data_collect_time = 0
        self.rl_train_time = 0
        self.all_time = 0

    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls, grad_norms):
        self.writer.add_scalar('performance/RLTrainFPS', self.agent_steps / self.rl_train_time, self.agent_steps)
        self.writer.add_scalar('performance/EnvStepFPS', self.agent_steps / self.data_collect_time, self.agent_steps)

        self.writer.add_scalar('losses/actor_loss', torch.mean(torch.stack(a_losses)).item(), self.agent_steps)
        self.writer.add_scalar('losses/bounds_loss', torch.mean(torch.stack(b_losses)).item(), self.agent_steps)
        self.writer.add_scalar('losses/critic_loss', torch.mean(torch.stack(c_losses)).item(), self.agent_steps)
        self.writer.add_scalar('losses/entropy', torch.mean(torch.stack(entropies)).item(), self.agent_steps)

        self.writer.add_scalar('info/last_lr', self.last_lr, self.agent_steps)
        self.writer.add_scalar('info/e_clip', self.e_clip, self.agent_steps)
        self.writer.add_scalar('info/kl', torch.mean(torch.stack(kls)).item(), self.agent_steps)
        self.writer.add_scalar('info/grad_norms', torch.mean(torch.stack(grad_norms)    ).item(), self.agent_steps)

        for k, v in self.extra_info.items():
            if isinstance(v, torch.Tensor) and len(v.shape) != 0:
                continue
            self.writer.add_scalar(f'{k}', v, self.agent_steps)

    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std.eval()
        if self.normalize_priv:
            self.priv_mean_std.eval()
        if self.normalize_point_cloud:
            self.point_cloud_mean_std.eval()
        if self.normalize_value:
            self.value_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_input:
            self.running_mean_std.train()
        if self.normalize_priv:
            self.priv_mean_std.train()
        if self.normalize_point_cloud:
            self.point_cloud_mean_std.train()
        if self.normalize_value:
            self.value_mean_std.train()

    def model_act(self, obs_dict):
        processed_obs = self.running_mean_std(obs_dict['obs'])
        priv_info = obs_dict['priv_info']
        if self.normalize_priv:
            priv_info = self.priv_mean_std(obs_dict['priv_info'])
            
        if self.normalize_point_cloud:
            point_cloud = self.point_cloud_mean_std(
                obs_dict['point_cloud_info'].reshape(-1, 3)
            ).reshape((processed_obs.shape[0], -1, 3))
        else:
            point_cloud = obs_dict['point_cloud_info']

        input_dict = {
            'obs': processed_obs,
            'priv_info': priv_info,
            'critic_info': obs_dict['critic_info'],
            'point_cloud_info': point_cloud,
            'proprio_hist': obs_dict['proprio_hist'],
            'tactile_hist': obs_dict['tactile_hist'],
            'obj_ends': obs_dict['obj_ends'],
        }
        res_dict = self.model.act(input_dict)
        res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        return res_dict

    def train(self):
        _t = time.time()
        _last_t = time.time()
        self.obs = self.env.reset()
        self.agent_steps = self.batch_size

        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            a_losses, c_losses, b_losses, entropies, kls, grad_norms = self.train_epoch()
            self.storage.data_dict = None

            for k, v in self.extra_info.items():
                # only log scalars
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    self.extra_info[k] = v

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()
            info_string = f'Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | ' \
                          f'Last FPS: {last_fps:.1f} | ' \
                          f'Collect Time: {self.data_collect_time / 60:.1f} min | ' \
                          f'Train RL Time: {self.rl_train_time / 60:.1f} min | ' \
                          f'Current Best: {self.best_rewards:.2f}'
            print(info_string)

            self.write_stats(a_losses, c_losses, b_losses, entropies, kls, grad_norms)
            mean_rewards = self.episode_rewards.get_mean()
            mean_lengths = self.episode_lengths.get_mean()
            self.writer.add_scalar('episode_rewards/step', mean_rewards, self.agent_steps)
            self.writer.add_scalar('episode_lengths/step', mean_lengths, self.agent_steps)
            checkpoint_name = f'ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}m_reward_{mean_rewards:.2f}'

            if self.save_freq > 0:
                if (self.epoch_num % self.save_freq == 0) and (mean_rewards <= self.best_rewards):
                    self.save(os.path.join(self.nn_dir, checkpoint_name))
                    self.save(os.path.join(self.nn_dir, f'last'))

            if mean_rewards > self.best_rewards and self.agent_steps >= self.save_best_after:
                print(f'save current best reward: {mean_rewards:.2f}')
                # remove previous best file
                prev_best_ckpt = os.path.join(self.nn_dir, f'best_reward_{self.best_rewards:.2f}.pth')
                if os.path.exists(prev_best_ckpt):
                    os.remove(prev_best_ckpt)
                self.best_rewards = mean_rewards
                self.save(os.path.join(self.nn_dir, f'best_reward_{mean_rewards:.2f}'))

        print('max steps achieved')

    def save(self, name):
        weights = {
            'model': self.model.state_dict(),
        }
        if self.running_mean_std:
            weights['running_mean_std'] = self.running_mean_std.state_dict()
        if self.normalize_priv:
            weights['priv_mean_std'] = self.priv_mean_std.state_dict()
        if self.normalize_point_cloud:
            weights['point_cloud_mean_std'] = self.point_cloud_mean_std.state_dict()
        if self.value_mean_std:
            weights['value_mean_std'] = self.value_mean_std.state_dict()
        torch.save(weights, f'{name}.pth')

    def restore_train(self, fn):
        if not fn:
            return
        print("loading checkpoint from path", fn)
        checkpoint = torch.load(fn)
        self.model.load_state_dict(checkpoint['model'])
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        if self.normalize_priv:
            self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])
        if self.normalize_point_cloud:
            self.point_cloud_mean_std.load_state_dict(checkpoint['point_cloud_mean_std'])

    def restore_test(self, fn):
        checkpoint = torch.load(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        if self.normalize_priv:
            self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])
        if self.normalize_point_cloud:
            self.point_cloud_mean_std.load_state_dict(checkpoint['point_cloud_mean_std'])

    def test(self):
        self.set_eval()
        obs_dict = self.env.reset()
        # import pickle
        # num_frames = 0
        while True:
            # with open("replay_round2030_obs.pkl", "ab") as f:
            #     pickle.dump(obs_dict['obs'], f)
            if not self.ppo_config.distill:
                if self.normalize_point_cloud:
                    point_cloud = self.point_cloud_mean_std(
                        obs_dict['point_cloud_info'].reshape(-1, 3)
                    ).reshape((obs_dict['obs'].shape[0], -1, 3))
                else:
                    point_cloud = obs_dict['point_cloud_info']
            if self.ppo_config.distill:
                assert NotImplementedError
                student_pc = obs_dict['student_pc_info']
                # temporary w/o one-hot
                # student_pc = self.point_cloud_mean_std(
                #             student_pc.reshape(-1, 6)[..., :3]
                #             ).reshape((obs_dict['obs'].shape[0], -1, 3))
                input_dict = {
                    'obs': self.running_mean_std(obs_dict['obs']),
                    'student_pc_info': student_pc,
                }
            else:
                input_dict = {
                    'obs': self.running_mean_std(obs_dict['obs']),
                    'priv_info': self.priv_mean_std(obs_dict['priv_info']) if self.normalize_priv else obs_dict['priv_info'],
                    'proprio_hist': obs_dict['proprio_hist'],
                    'point_cloud_info': point_cloud,
                }
            mu, extrin, extrin_gt = self.model.act_inference(input_dict)
            # assert extrin is not None

            mu = torch.clamp(mu, -1.0, 1.0)
            # print(mu)
            # with open("replay_round2030_action.pkl", "ab") as f:
            #     pickle.dump(mu, f)
            obs_dict, r, done, info = self.env.step(mu, extrin_record=extrin)
            # num_frames += 1
            # print(num_frames)
            # print(done.item())
            # if done.item():
            #     exit()

    def train_epoch(self):
        # collect minibatch data
        _t = time.time()
        self.set_eval()
        self.play_steps()
        self.data_collect_time += (time.time() - _t)
        # update network
        _t = time.time()
        self.set_train()
        a_losses, b_losses, c_losses = [], [], []
        entropies, kls, grad_norms = [], [], []
        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.storage)):
                value_preds, old_action_log_probs, advantage, old_mu, old_sigma, \
                    returns, actions, obs, priv_info, critic_info, point_cloud_info, proprio_hist, tactile_hist, obj_ends = self.storage[i]

                obs = self.running_mean_std(obs)
                if self.normalize_point_cloud:
                    point_cloud_info = self.point_cloud_mean_std(point_cloud_info.reshape(-1, 3)).reshape((obs.shape[0], -1, 3))

                batch_dict = {
                    'prev_actions': actions,
                    'obs': obs,
                    'priv_info': self.priv_mean_std(priv_info) if self.normalize_priv else priv_info,
                    'critic_info': critic_info,
                    'point_cloud_info': point_cloud_info,
                    'obj_ends': obj_ends,
                    'proprio_hist': proprio_hist,
                }
                res_dict = self.model(batch_dict)
                action_log_probs = res_dict['prev_neglogp']
                values = res_dict['values']
                entropy = res_dict['entropy']
                mu = res_dict['mus']
                sigma = res_dict['sigmas']

                # actor loss
                ratio = torch.exp(old_action_log_probs - action_log_probs)
                surr1 = advantage * ratio
                surr2 = advantage * torch.clamp(ratio, 1.0 - self.e_clip, 1.0 + self.e_clip)
                a_loss = torch.max(-surr1, -surr2)
                # critic loss
                value_pred_clipped = value_preds + (values - value_preds).clamp(-self.e_clip, self.e_clip)
                value_losses = (values - returns) ** 2
                value_losses_clipped = (value_pred_clipped - returns) ** 2
                c_loss = torch.max(value_losses, value_losses_clipped)
                # bounded loss
                if self.bounds_loss_coef > 0:
                    soft_bound = 1.1
                    mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
                    mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
                    b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
                else:
                    b_loss = torch.zeros_like(mu)
                a_loss, c_loss, entropy, b_loss = [torch.mean(loss) for loss in [a_loss, c_loss, entropy, b_loss]]
                loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
                self.optimizer.zero_grad()
                loss.backward()

                grad_norms.append(torch.norm(torch.cat([p.reshape(-1) for p in self.model.parameters()])))
                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)

                kl = kl_dist
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.storage.update_mu_sigma(mu.detach(), sigma.detach())

            av_kls = torch.mean(torch.stack(ep_kls))
            kls.append(av_kls)

            self.last_lr = self.scheduler.update(self.last_lr, av_kls.item())
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.last_lr

        self.rl_train_time += (time.time() - _t)
        return a_losses, c_losses, b_losses, entropies, kls, grad_norms

    def play_steps(self):
        for n in range(self.horizon_length):
            res_dict = self.model_act(self.obs)
            # collect o_t
            self.storage.update_data('obses', n, self.obs['obs'])
            self.storage.update_data('priv_info', n, self.obs['priv_info'])
            self.storage.update_data('critic_info', n, self.obs['critic_info'])
            self.storage.update_data('point_cloud_info', n, self.obs['point_cloud_info'])
            self.storage.update_data('proprio_hist', n, self.obs['proprio_hist'])
            self.storage.update_data('tactile_hist', n, self.obs['tactile_hist'])
            self.storage.update_data('obj_ends', n, self.obs['obj_ends'])
            for k in ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']:
                self.storage.update_data(k, n, res_dict[k])
            # do env step
            actions = torch.clamp(res_dict['actions'], -1.0, 1.0)

            # render() is called during env.step()
            # to save time, save gif only per gif_save_every_n steps
            # 1 step = #gpu * #envs agent steps
            record_frame = False
            if self.gif_frame_counter >= self.gif_save_every_n and self.gif_frame_counter % self.gif_save_every_n < self.gif_save_length:
                record_frame = True
            record_frame = record_frame and int(os.getenv('LOCAL_RANK', '0')) == 0
            self.env.enable_camera_sensors = record_frame
            self.gif_frame_counter += 1

            self.obs, rewards, self.dones, infos = self.env.step(actions)

            if record_frame and self.env.with_camera:
                self.gif_frames.append(self.env.capture_frame())
                # add frame to GIF
                if len(self.gif_frames) == self.gif_save_length:
                    frame_array = np.array(self.gif_frames)[None]  # add batch axis
                    self.writer.add_video(
                        'rollout_gif', frame_array, global_step=self.agent_steps,
                        dataformats='NTHWC', fps=20,
                    )
                    self.writer.flush()
                    self.gif_frames.clear()

            rewards = rewards.unsqueeze(1)
            # update dones and rewards after env step
            self.storage.update_data('dones', n, self.dones)
            rewards = rewards.to(self.device)
            shaped_rewards = 0.01 * rewards.clone()
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * infos['time_outs'].unsqueeze(1).float()
            self.storage.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_lengths += 1
            done_indices = self.dones.nonzero(as_tuple=False)
            self.episode_rewards.update(self.current_rewards[done_indices])
            self.episode_lengths.update(self.current_lengths[done_indices])

            assert isinstance(infos, dict), 'Info Should be a Dict'
            self.extra_info = infos

            not_dones = (1.0 - self.dones.float()).to(self.device)

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        res_dict = self.model_act(self.obs)
        last_values = res_dict['values']

        self.agent_steps = self.agent_steps + self.batch_size
        self.storage.computer_return(last_values, self.gamma, self.tau)
        self.storage.prepare_training()

        returns = self.storage.data_dict['returns']
        values = self.storage.data_dict['values']
        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()
        self.storage.data_dict['values'] = values
        self.storage.data_dict['returns'] = returns


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    c1 = torch.log(p1_sigma/p0_sigma + 1e-5)
    c2 = (p0_sigma ** 2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma ** 2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
    return kl.mean()


# from https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/algorithms/ppo.py
class AdaptiveScheduler(object):
    def __init__(self, kl_threshold=0.008):
        super().__init__()
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr
