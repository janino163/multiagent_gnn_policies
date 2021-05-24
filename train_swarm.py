from os import path
import configparser
import numpy as np
import random
import gym
# import gym_flock
import torch
import gym_swarm
import sys
import PIL.Image
# from learner.gnn_cloning import train_cloning
# from learner.gnn_dagger import train_dagger
# from learner.gnn_baseline import train_baseline
# from learner.gnn_baseline import train_baseline
from learner.gnn_swarm import train_swarm
def run_experiment(args, labels):
    # initialize gym env
    env_name = args.get('env')
    env = gym.make(env_name)

    # if isinstance(env.env, gym_swarm.envs.SwarmEnv):
    env.params_from_cfg(args)

    # use seed
    seed = args.getint('seed')
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # initialize params tuple
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    alg = args.get('alg').lower()
    if alg == 'swarm':
        stats = train_swarm(env, args, device, labels)
    else:
        raise Exception('Invalid algorithm/mode name')
    return stats


def main():
    fname = sys.argv[1]
    config_file = path.join(path.dirname(__file__), fname)
    config = configparser.ConfigParser()
    config.read(config_file)
    fname   = 'pic1.png'
    # max_size = 40
    # ref_img = PIL.Image.open(fname)
    # ref_img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
    # ref_img = 1.0-np.float32(ref_img)

    # # number_of_robots = int(ref_img.sum())
    # ref_img = torch.tensor(ref_img)
    # ref_array = torch.nonzero(ref_img).float()
    # ref_array = torch.swapaxes(ref_array,0,1)
    ref_array = torch.zeros([2, 100], dtype=torch.float32)
    printed_header = False

    if config.sections():
        for section_name in config.sections():
            if not printed_header:
                print(config[section_name].get('header'))
                printed_header = True

            stats = run_experiment(config[section_name], ref_array)
            print(section_name + ", " + str(stats['mean']) + ", " + str(stats['std']))
    else:
        val = run_experiment(config[config.default_section], ref_img)
        print(val)


if __name__ == "__main__":
    main()
