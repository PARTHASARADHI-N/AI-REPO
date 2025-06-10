from env import HighwayEnv, ACTION_NO_OP, get_highway_env
import numpy as np 
from typing import Tuple
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm 
import pandas as pd
import time
import imageio  # Added for GIF creation

class TabularQAgent:

    def __init__(self, 
                    env: HighwayEnv, 
                    alpha: float = 0.1, 
                    eps: float = 0.75, 
                    discount_factor: float = 0.9,
                    iterations: int = 100000, 
                    eps_type: str = 'constant',
                    validation_runs: int = 100,
                    validate_every: int = 1000,
                    visualize_runs: int = 10, 
                    visualize_every: int = 5000,
                    log_folder:str = './'
                    ):
        q_table = {}
        for speed in range(0,4):
            for lane in range(0,4):
                for d1 in range(0,5):
                    for d2 in range(0,5):
                        for d3 in range(0,5):
                            for d4 in range(0,5):
                                xt=(speed,lane,d1,d2,d3,d4)
                                q_table[xt]=np.array([0.0,0.0,0.0,0.0,0.0])
        self.df = discount_factor
        self.alpha = alpha
        self.env = env
        self.iterations = iterations
        self.validation_runs = validation_runs
        self.validate_every = validate_every
        self.visualization_runs = visualize_runs
        self.visualization_every = visualize_every
        self.log_folder = log_folder
        self.q_table = q_table
        self.eps = eps

    def choose_action(self, state, greedy = False):
        if (greedy):
            action=np.argmax(self.q_table[state])  
        else:
            if np.random.rand() < self.eps:
                action=np.random.randint(0,4)
            else:
                action=np.argmax(self.q_table[state])
        return action

    def validate_policy(self) -> Tuple[float, float]:
        rewards = []
        dist = []
        start_Q=[]
        for i in range(self.validation_runs):
            obs = self.env.reset(i)
            reward_accumulate=0
            terminated=False
            present_state=tuple(obs)
            value=max(self.q_table(present_state))
            gama=1
            count=0
            while not terminated:
                action=self.choose_action(present_state,True)
                nxt_state,reward,terminated,_=env.step(action) 
                reward_accumulate += gama * reward
                gama=gama*self.df
                distance_travelled = env.control_car.pos
                present_state = tuple(nxt_state)
                count+=1
           
            rewards.append(reward_accumulate)
            dist.append(distance_travelled) 
            start_Q.append(value)  
        print("average start state value",sum(start_Q)/len(start_Q))   
        return sum(rewards) / len(rewards), sum(dist) / len(dist)

    def visualize_policy(self, i: int) -> None:
        images=[]
        for j in range(self.visualization_runs):
            obs = self.env.reset(j)
            done = False
            present_state=tuple(obs)
            terminated=False
            while not terminated:
                image = self.env.render()
                images.append(image)
                action=self.choose_action(present_state,True)          
                nxt_state,reward,terminated,_=self.env.step(action)  
                present_state=tuple(nxt_state)
        plt.ion()
        fig, ax = plt.subplots()
        im = ax.imshow(images[0], cmap='gray')
        ax.set_title(f"Policy Visualization at iteration {i}") 
        for count, image in enumerate(images, start=1):
            im.set_data(image)  
            plt.draw() 
            plt.pause(0.2)  
        plt.ioff()
        plt.show(block=False)
        time.sleep(2)
        plt.close()

    def visualize_lane_value(self, i: int) -> None:
        '''
        Args:
            i: total iterations done so far
        
        Create image visualizations for no_op actions for particular lane
        '''
        for j in range(self.visualization_runs // 2):
            self.env.reset(j)  # don't modify this
            done = False
            k = 0
            images=[]
            
            while not done and k < 1000:  # Added upper limit to prevent infinite loops
                k += 1
                _, _, done, _ = self.env.step(ignore_control_car=True)
                
                if k % 20 == 0:
                    qvalues = []
                    states = self.env.get_all_lane_states()
                    
                    # Compute Q-values for "no op" action (assuming action 4 is no_op)
                    for state in states:
                        q_value = self.q_table.get(tuple(state), np.zeros(5))[4]
                        qvalues.append(q_value)
                    
                    # Average Q-values across speeds for each lane (assuming 4 speeds per lane)
                    lane_avg_values = []
                    for lane in range(4):
                        lane_qvalues = qvalues[lane * 4:(lane + 1) * 4]
                        avg = sum(lane_qvalues) / 4
                        lane_avg_values.extend([avg] * 4)
                    
                    # Render and save the visualization
                    image = self.env.render_lane_state_values(lane_avg_values)
                    images.append(image)
            imageio.mimsave(f'{self.log_folder}/lane_value{j}.gif', 
                          images, 
                          duration=0.1)

    def visualize_speed_value(self, i: int) -> None:
        '''
        Args:
            i: total iterations done so far
        
        Create image visualizations for no_op actions for particular lane
        '''
        for j in range(self.visualization_runs // 2):
            self.env.reset(j)  # don't modify this
            done = False
            k = 0
            Images=[]
            while not done and k < 1000:  # Added upper limit to prevent infinite loops
                k += 1
                _, _, done, _ = self.env.step(ignore_control_car=True)
                
                if k % 20 == 0:
                    qvalues = []
                    states = self.env.get_all_speed_states()
                    
                    # Compute Q-values for "no op" action (assuming action 4 is no_op)
                    for state in states:
                        q_value = self.q_table.get(tuple(state), np.zeros(5))[4]
                        qvalues.append(q_value)
                    
                    # Render and save the visualization
                    image = self.env.render_speed_state_values(qvalues)
                    Images.append(image)
            imageio.mimsave(f'{self.log_folder}/speed_value{j}.gif', 
                          Images, 
                          duration=0.1)
    def get_policy(self):
        avg_rewards=[]
        max_distances=[]
        for i in tqdm(range(self.iterations), desc = "Progress", leave=True):
            terminated = False
            present_state=tuple(env.reset(i))
    
            while(terminated == False):
                action = self.choose_action(present_state)
                nxt_state,reward,terminated,_=env.step(action)
                image = env.render()
                nxt_state = tuple(nxt_state)
                self.q_table[present_state][action]+=self.alpha*(reward+self.df*np.max(self.q_table[nxt_state])-self.q_table[present_state][action])
                if terminated:
                    break
                else:
                    present_state=nxt_state
            if (i+1)%self.validate_every==0:
                avg_reward,avg_distance = self.validate_policy()
                print(f"Iteration {i+1}: Avg Reward = {avg_reward:.4f}, Avg Distance = {avg_distance:.4f}")

                avg_rewards.append(avg_reward)
                max_distances.append(avg_distance)
            if (i+1)%self.visualization_every==0:
                # self.visualize_policy(i)
                pass

        # Generate 10 GIFs at the end of training
        self.visualize_lane_value(i)
        self.visualize_speed_value(i)
        for traj in range(10):
            trajectory_images = []
            obs = self.env.reset(traj)
            terminated = False
            present_state = tuple(obs)
            while not terminated:
                image = self.env.render()
                trajectory_images.append(image)
                action = self.choose_action(present_state, True)
                nxt_state, reward, terminated, _ = self.env.step(action)
                present_state = tuple(nxt_state)
            # Save GIF
            imageio.mimsave(f'{self.log_folder}/trajectory_{traj}.gif', 
                          trajectory_images, 
                          duration=0.1)

        df = pd.DataFrame.from_dict(self.q_table, orient="index")
        df.to_csv(f"{self.log_folder}/Q_table.csv")
                        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(range(0, self.iterations, self.validate_every), avg_rewards)
        plt.title('Discounted Return vs Training Iterations')
        plt.xlabel('Training Iterations')
        plt.ylabel('Average Discounted Return')
        plt.savefig(f'{self.log_folder}/discounted_return.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(range(0, self.iterations, self.validate_every), max_distances)
        plt.title('Maximum Distance vs Training Iterations')
        plt.xlabel('Training Iterations')
        plt.ylabel('Average Maximum Distance')
        plt.savefig(f'{self.log_folder}/max_distance.png')
        plt.close()

        return self.q_table

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse command-line arguments.")
    parser.add_argument("--iterations", type=int, required=True, help="Number of iterations (integer).")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the input file.")
    args = parser.parse_args()
    
    env = get_highway_env(dist_obs_states = 5, reward_type = 'dist')
    env = HighwayEnv()
    qagent = TabularQAgent(env, 
                          iterations=args.iterations,
                          log_folder = args.output_folder)
    qagent.get_policy()