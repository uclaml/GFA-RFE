import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from agent.mddpg import MDDPGAgent

class RFE(nn.Module):
    def __init__(self):
        super().__init__()

class RFEAgent(MDDPGAgent):

    def __init__(self, update_encoder, **kwargs):
        super().__init__(**kwargs)


    def compute_intr_reward(self, obs, action, discount, next_obs, step):
        
        
        # reward = torch.div(torch.std(self.critic(obs, action).detach(), dim=0) , torch.mean(torch.abs(self.critic(obs, action).detach()), dim=0))
        reward = torch.std(self.critic(obs, action).detach(), dim=0) * (1 - discount)
        # reward = torch.std(self.critic(obs, action).detach(), dim=0)
        reward = reward.reshape(-1, 1)
        
        # print("reward: ", torch.mean(reward).item())
        return reward


    def update(self, replay_iter, step):
        metrics = dict()
        #import ipdb; ipdb.set_trace()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, extr_reward, discount, next_obs = utils.to_torch(batch, self.device)


        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.reward_free:

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(obs, action, discount, next_obs, step)

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()

            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics