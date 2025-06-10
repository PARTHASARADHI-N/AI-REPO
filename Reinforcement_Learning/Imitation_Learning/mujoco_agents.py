
import itertools
import torch
import random
from torch import nn
from torch import optim
import numpy as np
from tqdm import tqdm
import torch.distributions as distributions

from utils.replay_buffer import ReplayBuffer
import utils.utils as utils
from agents.base_agent import BaseAgent
import utils.pytorch_util as ptu
from policies.experts import load_expert_policy
from tqdm import  tqdm
from time import time

class ImitationAgent(BaseAgent):
    def __init__(self, observation_dim:int, action_dim:int, args = None, discrete:bool = False, **hyperparameters ):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.action_dim  = action_dim
        self.observation_dim = observation_dim
        self.is_action_discrete = discrete
        self.args = args
        self.replay_buffer = ReplayBuffer(100000)
        self.beta = 0.99
        self.best_reward=-9999
        self.model = ptu.build_mlp(self.observation_dim, self.action_dim, n_layers=self.hyperparameters["n_layers"], size=self.hyperparameters["hidden_size"])
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.loss_values=[]
        self.t1=time()
   

    def forward(self, observation: torch.FloatTensor):
       
        if random.random() < self.beta:
            expert_a = self.expert_policy.get_action(observation)
            expert_a = (torch.tensor(expert_a)
                        if not torch.is_tensor(expert_a)
                        else expert_a)
            return expert_a
        else:
            return self.model(observation)
        
        
    @torch.no_grad()
    def get_action(self, observation: torch.FloatTensor):
        return self.model(observation)

    def update(self, observations, actions):
        self.optimizer.zero_grad()
        predicted_actions = self.model(observations)
        loss = nn.MSELoss()(predicted_actions, actions)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_iteration(self, env, envsteps_so_far, render=False, itr_num=None, **kwargs):
        if not hasattr(self, "expert_policy"):
            self.expert_policy, initial_expert_data = load_expert_policy(env, self.args.env_name)
            self.replay_buffer.add_rollouts(initial_expert_data)

        k=self.replay_buffer.__len__()

        sample = self.replay_buffer.sample_batch(k)
        i=0
        running_loss=[]
        while i<k-64:
            j=i+64
            obs_batch = torch.FloatTensor(sample['obs'][i:j])
            action_batch = torch.FloatTensor(sample['acs'][i:j])
            loss = self.update(obs_batch, action_batch)
            running_loss.append(loss)
            i=j


        self.loss_values.append(sum(running_loss))
        self.beta = self.beta**(itr_num)
        rollouts = []
        n=3
        for j in range(n):
            episode=[]
            obs = env.reset()
            done=False
            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action = self.forward(obs_tensor).squeeze(0).detach().cpu().numpy()
                next_obs, reward, done, _ = env.step(action)
                expert_action = self.expert_policy.get_action(torch.FloatTensor(obs))
                
                traj = {
                    "observation": np.array(obs)[None, :],
                    "action": np.array(expert_action)[None, :],
                    "next_observation": np.array(next_obs)[None, :],
                    "reward": np.array([reward]),
                    "terminal": np.array([done])
                }
                obs = next_obs if not done else env.reset()
                episode.append(traj)
               
            rollouts.append(traj)
            
        self.replay_buffer.add_rollouts(rollouts)

        env_type=self.hyperparameters["env_type"]
    
        eval_trajs, eval_envsteps_this_batch = utils.sample_trajectories(
            env, self.get_action, 15* 1000, 1000
        )

        logs = utils.compute_metrics(rollouts, eval_trajs)
        logs["Eval_AverageReturn"]
        if  logs["Eval_AverageReturn"] >self.best_reward:
                self.best_reward=logs["Eval_AverageReturn"]
                torch.save(self.model.state_dict(),f"best_models/{env_type}.pth")

        return {
            'episode_loss': loss,
            'trajectories': rollouts,
            'current_train_envsteps': envsteps_so_far
        }
