from env import HighwayEnv, ACTION_NO_OP, get_highway_env
import numpy as np
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import time
import imageio
import os

random.seed(8)
torch.random.manual_seed(8)

# Neural Network for Q-Function
class DQN(nn.Module):
    def __init__(self, state_dim=6, action_dim=5):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(state_dim, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.l3(x)
        return x

# Replay Buffer Implementation
class ReplayBuffer:
    def __init__(self, max_size, state_dim):
        self.max_size = max_size
        # Initialize storage as lists instead of tensors
        self.states = [None] * max_size
        self.actions = [None] * max_size
        self.rewards = [None] * max_size
        self.next_states = [None] * max_size
        self.terminates = [None] * max_size
        self.ptr = 0
        self.size = 0
        self.state_dim = state_dim

    def add(self, state, action, reward, next_state, terminated):
        # Store raw values without conversion to tensors
        self.states[self.ptr] = state  # Expecting state as a NumPy array or list
        self.actions[self.ptr] = action  # Expecting action as an integer
        self.rewards[self.ptr] = reward  # Expecting reward as a float
        self.next_states[self.ptr] = next_state  # Expecting next_state as a NumPy array or list
        self.terminates[self.ptr] = terminated  # Expecting terminated as a boolean
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,))
        # Convert sampled data to tensors
        states = torch.tensor([self.states[i] for i in indices], dtype=torch.float32)
        actions = torch.tensor([self.actions[i] for i in indices], dtype=torch.int64)
        rewards = torch.tensor([self.rewards[i] for i in indices], dtype=torch.float32)
        next_states = torch.tensor([self.next_states[i] for i in indices], dtype=torch.float32)
        terminates = torch.tensor([self.terminates[i] for i in indices], dtype=torch.bool)
        return states, actions, rewards, next_states, terminates
# DQN Agent
class DQNAgent:
    def __init__(self,
                 env: HighwayEnv,
                 alpha: float = 0.1,
                 eps: float = 0.75,
                 discount_factor: float = 0.99,
                 tau: float = 0.005,
                 iterations: int = 100000,
                 eps_type: str = 'constant',
                 validation_runs: int = 1000,
                 validate_every: int = 50000,
                 visualize_runs: int = 10,
                 visualize_every: int = 50000,
                 log_folder: str = './',
                 lr: float = 0.0001,
                 batch_size: int = 256,
                 buffer_size: int = 100000):
        self.env = env
        self.eps = eps
        self.df = discount_factor
        self.alpha = alpha
        self.iterations = iterations
        self.validation_runs = validation_runs
        self.validate_every = validate_every
        self.visualization_runs = visualize_runs
        self.visualization_every = visualize_every
        self.log_folder = log_folder
        self.lr = lr
        self.batch_size = batch_size
        self.tau = tau

        # Device setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Network setup
        state_dim = 6  # Matches env state size
        action_dim = 5  # Matches env action size
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.buffer = ReplayBuffer(max_size=buffer_size, state_dim=state_dim)

    def choose_action(self, state, greedy=False):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        if greedy or random.random() > self.eps:
            with torch.no_grad():
                q_values = self.model(state_tensor)
            action = torch.argmax(q_values).item()
        else:
            action = random.randint(0, 4)  # 5 actions: 0 to 4
        return action

    def validate_policy(self) -> Tuple[float, float]:
        rewards = []
        dist = []
        for i in range(self.validation_runs):
            obs = self.env.reset(i)
            terminated = False
            total_reward = 0
            gamma = 1.0
            while not terminated:
                action = self.choose_action(obs, greedy=True)
                nxt_state, reward, terminated, _ = self.env.step(action)
                total_reward += gamma * reward
                gamma *= self.df
                obs = nxt_state
            rewards.append(total_reward)
            dist.append(self.env.control_car.pos)
        avg_reward = sum(rewards) / len(rewards)
        avg_dist = sum(dist) / len(dist)
        return avg_reward, avg_dist

    def visualize_policy(self, i: int) -> None:
        for j in range(self.visualization_runs):
            obs = self.env.reset(j)
            terminated = False
            images = []
            while not terminated:
                image = self.env.render()
                images.append(image)
                action = self.choose_action(obs, greedy=True)
                nxt_state, _, terminated, _ = self.env.step(action)
                obs = nxt_state
            # Save as GIF
            gif_path = f'{self.log_folder}/policy_iter{i}_run{j}.gif'
            imageio.mimsave(gif_path, images, duration=0.1)

    def visualize_lane_value(self, i: int) -> None:
        for j in range(self.visualization_runs // 2):
            self.env.reset(j)
            done = False
            k = 0
            Images=[]
            while not done and k < 1000:  # Prevent infinite loops
                k += 1
                _, _, done, _ = self.env.step(ignore_control_car=True)
                if k % 20 == 0:
                    states = self.env.get_all_lane_states()
                    qvalues = []
                    for state in states:
                        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
                        with torch.no_grad():
                            q_values = self.model(state_tensor)
                        qvalues.append(q_values[ACTION_NO_OP].item())
                    # Average across speeds for each lane
                    lane_avg_values = []
                    for lane in range(4):
                        lane_qvalues = qvalues[lane * 4:(lane + 1) * 4]
                        avg = sum(lane_qvalues) / 4
                        lane_avg_values.extend([avg] * 4)
                    image = self.env.render_lane_state_values(lane_avg_values)
                    Images.append(image)
            imageio.mimsave(f'{self.log_folder}/lane_value{j}.gif', 
                          Images, 
                          duration=0.1)
                

    def visualize_speed_value(self, i: int) -> None:
        for j in range(self.visualization_runs // 2):
            self.env.reset(j)
            done = False
            k = 0
            Images=[]
            while not done and k < 1000:  # Prevent infinite loops
                k += 1
                _, _, done, _ = self.env.step(ignore_control_car=True)
                if k % 20 == 0:
                    states = self.env.get_all_speed_states()
                    qvalues = []
                    for state in states:
                        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
                        with torch.no_grad():
                            q_values = self.model(state_tensor)
                        qvalues.append(q_values[ACTION_NO_OP].item())
                    image = self.env.render_speed_state_values(qvalues)
                    Images.append(image)
            imageio.mimsave(f'{self.log_folder}/speed_value{j}.gif', 
                          Images, 
                          duration=0.1)

    def get_policy(self):
        avg_rewards = []
        max_distances = []
        update_frequency = 20
        for i in tqdm(range(self.iterations), desc='Training', leave=True):
            present_state = self.env.reset(i)
            terminated = False
            step_count = 0
            while not terminated:
                action = self.choose_action(present_state)
                nxt_state, reward, terminated, _ = self.env.step(action)
                self.buffer.add(present_state, action, reward, nxt_state, terminated)
                step_count+=1
                present_state=nxt_state
                if self.buffer.size > self.batch_size and step_count % update_frequency == 0:
                        states, actions, rewards, next_states, terminates = self.buffer.sample(self.batch_size)
                        states = states.to(self.device)
                        actions = actions.to(self.device)
                        rewards = rewards.to(self.device)
                        next_states = next_states.to(self.device)
                        terminates = terminates.to(self.device)

                        # Compute targets
                        with torch.no_grad():
                            next_q_values = self.target_model(next_states).max(1)[0]
                            targets = rewards + self.df * (1 - terminates.float()) * next_q_values

                        # Compute current Q-values
                        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                        # Compute loss
                        loss = self.loss_fn(q_values, targets)

                        # Optimize
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()    

            # Soft update target network
            if (i + 1) % 100 == 0:
                for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

            # Validation
            if (i + 1) % self.validate_every == 0:
                avg_reward, avg_dist = self.validate_policy()
                avg_rewards.append(avg_reward)
                max_distances.append(avg_dist)
                print(f"Iteration {i+1}: Avg Reward = {avg_reward:.4f}, Avg Distance = {avg_dist:.4f}")

            # Visualization
            if (i + 1) % self.visualization_every == 0:
                # self.visualize_policy(i)
                # self.visualize_lane_value(i)
                # self.visualize_speed_value(i)
                pass

  

        # Save model
        torch.save(self.model.state_dict(), os.path.join(self.log_folder, "trained_model.pth"))
        self.visualize_lane_value(i)
        self.visualize_policy(i)
        self.visualize_speed_value(i)
        # Plot results
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(0, self.iterations, self.validate_every), np.array(avg_rewards).flatten(), label="Avg Reward")
        plt.xlabel("Iterations")
        plt.ylabel("Average Reward")
        plt.title("Validation Reward over Iterations")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(0, self.iterations, self.validate_every), np.array(max_distances).flatten(), label="Avg Distance")
        plt.xlabel("Iterations")
        plt.ylabel("Average Distance")
        plt.title("Validation Distance over Iterations")
        plt.legend()

        plt.savefig(os.path.join(self.log_folder, "validation_plots.png"))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse command-line arguments.")
    parser.add_argument("--iterations", type=int, required=True, help="Number of iterations (integer).")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder.")
    args = parser.parse_args()

    # Part 1: Discrete state representation
    env = get_highway_env(dist_obs_states=5, reward_type='dist')
    # Part 2: Continuous state representation (uncomment below and comment above)
    # env = get_highway_env(dist_obs_states=5, reward_type='dist', obs_type='continuous')

    qagent = DQNAgent(
        env,
        iterations=args.iterations,
        log_folder=args.output_folder
    )
    qagent.get_policy()