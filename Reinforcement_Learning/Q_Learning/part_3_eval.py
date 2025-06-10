from part_3 import BestAgent 
import argparse
from tqdm import tqdm

def evaluate_episode(agent, seed):

    obs = agent.env.reset(seed) #don't modify this
    done = False
    while(not done):
        a = agent.choose_action(obs)
        obs, _, done, _ = agent.env.step(a)
    return agent.env.control_car.pos

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Parse command-line arguments.")
    parser.add_argument("--iterations", type=int, required=True, help="Number of iterations (integer).")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file.")
    args = parser.parse_args()

    agent = BestAgent(iterations = args.iterations)
    agent.get_policy()

    file = open(args.input_file, "r")
    seeds = file.readlines()
    seeds = [int(s.strip()) for s in seeds]
    
    #evaluate
    distances = []
    for s in tqdm(seeds):
        distances.append(evaluate_episode(agent, s))

    file = open(args.output_file, "w")
    for d in distances:
        file.write(str(d) + '\n')