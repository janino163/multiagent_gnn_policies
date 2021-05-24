import numpy as np
import os

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable

from learner.state_with_delay import MultiAgentStateWithDelay
from learner.replay_buffer import ReplayBuffer
from learner.replay_buffer import Transition
from learner.actor import Actor

#
# # TODO: how to deal with bounded/unbounded action spaces?? Should I always assume bounded actions?


class Swarm(object):

    def __init__(self, device, args, k=None):  # , n_s, n_a, k, device, hidden_size=32, gamma=0.99, tau=0.5):
        """
        Initialize the DDPG networks.
        :param device: CUDA device for torch
        :param args: experiment arguments
        """

        n_s = args.getint('n_states')
        n_a = args.getint('n_actions')
        k = k or args.getint('k')
        hidden_size = args.getint('hidden_size')
        n_layers = args.getint('n_layers') or 2
        gamma = args.getfloat('gamma')
        tau = args.getfloat('tau')
        self.dt = args.getfloat('dt')

        self.n_agents = args.getint('n_agents')
        self.n_states = n_s
        self.n_actions = n_a

        # Device
        self.device = device

        hidden_layers = [hidden_size] * n_layers
        ind_agg = 0  # int(len(hidden_layers) / 2)  # aggregate halfway

        # Define Networks
        self.actor = Actor(n_s, n_a, hidden_layers, k, ind_agg).to(self.device)

        # Define Optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=args.getfloat('actor_lr'))

        # Constants
        self.gamma = gamma
        self.tau = tau

    def select_action(self, state):
        """
        Evaluate the Actor network over the given state, and with injection of noise.
        :param state: The current state.
        :param graph_shift_op: History of graph shift operators
        :param action_noise: The action noise
        :return:
        """
        self.actor.eval()  # Switch the actor network to Evaluation Mode.
        mu = self.actor(state.delay_state, state.delay_gso)  # .to(self.device)

        # mu is (B, 1, nA, N), need (N, nA)
        mu = mu.permute(0, 1, 3, 2)
        mu = mu.view((self.n_agents, self.n_actions))

        self.actor.train()  # Switch back to Train mode.
        mu = mu.data
        return mu

        # return mu.clamp(-1, 1)  # TODO clamp action to what space?
    def integrate_control(self, x, u):
        # x position
        # print(x.shape)
        # print(u.shape)
        # exit()
        x_ = x[:, 0, :]
        y_ = x[:, 1, :]
        vx = x[:, 2, :]
        vy = x[:, 3, :]
        ax = u[:, 0, :]
        ay = u[:, 1, :]
        # x positon
        d1 = x_ + vx * self.dt + ax * self.dt * self.dt * 0.5
        # y position
        d2 = y_ + vy * self.dt + ay * self.dt * self.dt * 0.5
        # x velocity
        d3 = vx + ax * self.dt
        # y velocity
        d4 = vy + ay * self.dt
        # print(torch.stack([d1, d2, d3, d4], dim=1).shape)
        # exit()
        return torch.stack([d1, d2, d3, d4], dim=1)

    def gradient_step(self, batch, labels=None):
        """
        Take a gradient step given a batch of sampled transitions.
        :param batch: The batch of training samples.
        :return: The loss function in the network.
        """

        delay_gso_batch = Variable(torch.cat(tuple([s.delay_gso for s in batch.state]))).to(self.device)
        delay_state_batch = Variable(torch.cat(tuple([s.delay_state for s in batch.state]))).to(self.device)
        actor_batch = self.actor(delay_state_batch, delay_gso_batch) #(B, 1, nA, N)
        optimal_action_batch = Variable(torch.cat(batch.action)).to(self.device)
        # print(delay_state_batch[:, 0, 0:2, :].shape)
        # print(labels[:, 0:100].repeat(20,1,1).shape)
        # print(actor_batch.shape)
        # exit()
        control_label = torch.zeros(actor_batch.shape, dtype=torch.float32)
        x = self.integrate_control(delay_state_batch[:, 0, 0:4, :], actor_batch[:,0,:,:])
        # print(optimal_action_batch)
        # exit()
        # position = delay_state_batch +
        # Optimize Actor
        self.actor_optim.zero_grad()

        # Loss related to sampled Actor Gradient.
        policy_loss = F.mse_loss(x[:,0:2,:], labels.repeat(20,1,1).to(self.device))
        policy_loss = policy_loss + F.mse_loss(actor_batch, control_label.to(self.device))
        policy_loss.backward()
        self.actor_optim.step()
        # print('')
        # End Optimize Actor

        return policy_loss.item()

    def save_model(self, env_name, suffix="", actor_path=None):
        """
        Save the Actor Model after training is completed.
        :param env_name: The environment name.
        :param suffix: The optional suffix.
        :param actor_path: The path to save the actor.
        :return: None
        """
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/actor_{}_{}".format(env_name, suffix)
        print('Saving model to {}'.format(actor_path))
        torch.save(self.actor.state_dict(), actor_path)

    def load_model(self, actor_path, map_location):
        """
        Load Actor Model from given paths.
        :param actor_path: The actor path.
        :return: None
        """
        # print('Loading model from {}'.format(actor_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path, map_location))
            self.actor.to(self.device)


def train_swarm(env, args, device, label=None):
    x_ = torch.tensor([0,1,2,3,4,5,6,7,8,9], dtype=torch.float32) * 0.5
    y_ = torch.tensor([0,1,2,3,4,5,6,7,8,9], dtype=torch.float32) * 0.5
    gridx, gridy = torch.meshgrid(x_, y_)
    gridx = gridx.flatten()
    gridy = gridy.flatten()



    label[0,:] = gridx + 10.0
    label[1,:] = gridy + 10.0
    debug = args.getboolean('debug')
    memory = ReplayBuffer(max_size=args.getint('buffer_size'))
    learner = Swarm(device, args)

    n_a = args.getint('n_actions')
    n_agents = args.getint('n_agents')
    batch_size = args.getint('batch_size')

    n_train_episodes = args.getint('n_train_episodes')
    beta_coeff = args.getfloat('beta_coeff')
    test_interval = args.getint('test_interval')
    n_test_episodes = args.getint('n_test_episodes')

    total_numsteps = 0
    updates = 0
    beta = 1

    stats = {'mean': -1.0 * np.Inf, 'std': 0}

    for i in range(n_train_episodes):

        beta = max(beta * beta_coeff, 0.5)

        state = MultiAgentStateWithDelay(device, args, env.reset(), prev_state=None)

        done = False
        policy_loss_sum = 0
        while not done:
            optimal_action = env.controller()

            if np.random.binomial(1, beta) > 0:
                action = optimal_action
            else:
                action = learner.select_action(state)
                action = action.cpu().numpy()

            next_state, reward, done, _ = env.step(action)

            next_state = MultiAgentStateWithDelay(device, args, next_state, prev_state=state)

            total_numsteps += 1

            # action = torch.Tensor(action)
            notdone = torch.Tensor([not done]).to(device)
            reward = torch.Tensor([reward]).to(device)

            # action is (N, nA), need (B, 1, nA, N)
            optimal_action = torch.Tensor(optimal_action).to(device)
            optimal_action = optimal_action.transpose(1, 0)
            optimal_action = optimal_action.reshape((1, 1, n_a, n_agents))

            memory.insert(Transition(state, optimal_action, notdone, next_state, reward))

            state = next_state

        if memory.curr_size > batch_size:

            # labels = 
            for _ in range(args.getint('updates_per_step')):
                transitions = memory.sample(batch_size)
                batch = Transition(*zip(*transitions))
                policy_loss = learner.gradient_step(batch, label)
                policy_loss_sum += policy_loss
                updates += 1

        if i % test_interval == 0 and debug:
            test_rewards = []
            for j_ in range(n_test_episodes):
                ep_reward = 0
                state = MultiAgentStateWithDelay(device, args, env.reset(), prev_state=None)
                done = False
                while not done:
                    action = learner.select_action(state)
                    next_state, reward, done, _ = env.step(action.cpu().numpy())
                    next_state = MultiAgentStateWithDelay(device, args, next_state, prev_state=state)
                    ep_reward += reward
                    state = next_state
                    env.render(targets=label, name='test_'+str(int(i / test_interval))+'_step_' + str(j_))
                test_rewards.append(ep_reward)

            mean_reward = np.mean(test_rewards)
            # if stats['mean'] < mean_reward:
            #     stats['mean'] = mean_reward
            #     stats['std'] = np.std(test_rewards)
            #
            #     if debug and args.get('fname'):  # save the best model
            #         learner.save_model(args.get('env'), suffix=args.get('fname'))

            if debug:
                print(
                    "Episode: {}, updates: {}, total numsteps: {}, reward: {}, policy loss: {}".format(
                        i, updates,
                        total_numsteps,
                        mean_reward,
                        policy_loss_sum))

    test_rewards = []
    for _ in range(n_test_episodes):
        ep_reward = 0
        state = MultiAgentStateWithDelay(device, args, env.reset(), prev_state=None)
        done = False
        while not done:
            action = learner.select_action(state)
            next_state, reward, done, _ = env.step(action.cpu().numpy())
            next_state = MultiAgentStateWithDelay(device, args, next_state, prev_state=state)
            ep_reward += reward
            state = next_state
            env.render()
        test_rewards.append(ep_reward)

    mean_reward = np.mean(test_rewards)
    stats['mean'] = mean_reward
    stats['std'] = np.std(test_rewards)

    if debug and args.get('fname'):  # save the best model
        learner.save_model(args.get('env'), suffix=args.get('fname'))

    env.close()
    return stats
