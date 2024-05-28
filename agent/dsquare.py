import math
import warnings
from logging import warn

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from agent.ddpg import Actor, DDPGAgent, Encoder


class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x @ self.weight + self.bias
    
class VectorizedCritics(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim, ensemble_size):
        super().__init__()

        self.obs_type = obs_type
        
        if obs_type == 'pixels':
            self.trunk = nn.Sequential(
                VectorizedLinear(obs_dim, feature_dim, ensemble_size),
                nn.LayerNorm(feature_dim),
                nn.Tanh()
            )
            self.Qs = nn.Sequential(
                VectorizedLinear(feature_dim + action_dim, hidden_dim, ensemble_size),
                nn.ReLU(inplace=True),
                VectorizedLinear(hidden_dim, hidden_dim, ensemble_size),
                nn.ReLU(inplace=True),
                VectorizedLinear(hidden_dim, 1, ensemble_size)
            )
        else:
            self.trunk = nn.Sequential(
                VectorizedLinear(obs_dim + action_dim, hidden_dim, ensemble_size),
                nn.LayerNorm(hidden_dim),  
                nn.Tanh()
            )
            self.Qs = nn.Sequential(
                VectorizedLinear(hidden_dim, hidden_dim, ensemble_size),
                nn.ReLU(inplace=True),
                VectorizedLinear(hidden_dim, 1, ensemble_size)
            )

    def forward(self, obs, action):
        inpt = obs if self.obs_type == 'pixels' else torch.cat([obs, action],dim=-1)
        h = self.trunk(inpt)
        h = torch.cat([h, action], dim=-1) if self.obs_type == 'pixels' else h
        Qs = self.Qs(h)
        return Qs
    
class D2Agent(DDPGAgent):
    
    def __init__(self,
                 name,
                 reward_free,
                 obs_type,
                 obs_shape,
                 action_shape,
                 device,
                 lr,
                 feature_dim,
                 hidden_dim,
                 ensemble_size,
                 critic_target_tau,
                 num_expl_steps,
                 update_every_steps,
                 stddev_schedule,
                 nstep,
                 batch_size,
                 stddev_clip,
                 init_critic,
                 use_tb,
                 use_wandb,
                 greedy_epsilon,
                 update_encoder,
                 ucb,
                 noise_coe,
                 weight,
                 weight_offline,
                 weight_lo,
                 weight_up,
                 meta_dim=0):
        self.reward_free = reward_free
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.init_critic = init_critic
        self.feature_dim = feature_dim
        self.solved_meta = None
        self.ensemble_size = ensemble_size
        self.update_encoder = update_encoder
        self.greedy_epsilon = greedy_epsilon
        self.ucb = ucb
        self.noise_coe = noise_coe
        self.weight = weight
        self.weight_offline = weight_offline
        self.weight_lo = weight_lo
        self.weight_up = weight_up
        # models
        if obs_type == 'pixels':
            self.aug = utils.RandomShiftsAug(pad=4)
            self.encoder = Encoder(obs_shape).to(device)
            self.obs_dim = self.encoder.repr_dim + meta_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = obs_shape[0] + meta_dim

        self.actor = Actor(obs_type, self.obs_dim, self.action_dim,
                           feature_dim, hidden_dim).to(device)

        self.critic = VectorizedCritics(obs_type, self.obs_dim, self.action_dim, 
                                       feature_dim, hidden_dim, ensemble_size).to(device)

        self.critic_target = VectorizedCritics(obs_type, self.obs_dim, 
                                              self.action_dim, feature_dim, hidden_dim, ensemble_size).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        


        if obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(),
                                                lr=lr)
        else:
            self.encoder_opt = None
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.train()
        self.critic_target.train()
        
        
    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()
        
        if self.reward_free == True:
            with torch.no_grad():
                stddev = utils.schedule(self.stddev_schedule, step)
                dist = self.actor(next_obs, stddev)
                next_action = dist.sample(clip=self.stddev_clip)
                target_Qs = self.critic_target(next_obs, next_action)
                target_V = target_Qs.mean(dim=0)
                target_Q = reward + (discount * target_V)
                target_Q.clamp_(0, 500)
                std = target_Qs.std(dim=0).clamp_(self.weight_lo, self.weight_up) # change this clamp value if needed
            Qs = self.critic(obs, action)
            if self.weight:
                critic_loss = F.mse_loss(Qs[(step // self.update_every_steps) % self.ensemble_size] / std, target_Q / std)
            else:
                critic_loss = F.mse_loss(Qs[(step // self.update_every_steps) % self.ensemble_size], target_Q)
            
            if self.use_tb or self.use_wandb:
                metrics['critic_target_q'] = target_Q.mean().item()
                metrics['critic_q1'] = target_Qs.min(dim=0).values.mean().item()
                metrics['critic_q2'] = target_Qs.max(dim=0).values.mean().item()
                metrics['critic_loss'] = critic_loss.item()
        
        else:
            with torch.no_grad():
                stddev = utils.schedule(self.stddev_schedule, step)
                dist = self.actor(next_obs, stddev)
                next_action = dist.sample(clip=self.stddev_clip)
                next_Qs = self.critic_target(next_obs, next_action)
                rewards = torch.repeat_interleave(reward.unsqueeze(dim=0), self.ensemble_size, dim=0)
                target_Qs = rewards + (discount * next_Qs)
                std = target_Qs.std(dim=0).clamp_(self.weight_lo, self.weight_up)
            Qs = self.critic(obs, action)

            if self.weight_offline:
                critic_loss =  2 * F.mse_loss(Qs / std, target_Qs / std)
            else:
                critic_loss =  2 * F.mse_loss(Qs, target_Qs)
            
            if self.use_tb or self.use_wandb:
                metrics['critic_q1'] = target_Qs.min(dim=0).values.mean().item()
                metrics['critic_q2'] = target_Qs.max(dim=0).values.mean().item()
                metrics['critic_loss'] = critic_loss.item()
            
        

        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        return metrics

    
    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Qs = self.critic(obs, action)
        Q = torch.mean(Qs, 0)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, extr_reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.reward_free:
            with torch.no_grad():
                intr_reward = self.compute_intr_reward(obs, action, discount)

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # update critic
        metrics.update(self.update_critic(obs.detach(), action, reward, discount, next_obs.detach(), step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau / self.ensemble_size)


        return metrics

    def compute_intr_reward(self, obs, action, discount):
        
        reward = torch.std(self.critic_target(obs, action), dim=0) * self.ucb
        if self.noise_coe != 0:
            reward += self.noise_coe * torch.randn(reward.shape).to(self.device)
        reward = reward.reshape(-1, 1).clamp_(0, 10)
        
        return reward