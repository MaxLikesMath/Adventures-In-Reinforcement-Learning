import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Categorical

seed=123
env = gym.make("LunarLander-v2")
#env = gym.make("CartPole-v0")
env.seed(seed)
torch.manual_seed(seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n

        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128, self.action_space, bias=False)

        self.gamma = gamma

        # Episode policy and reward history
        self.policy_history = torch.autograd.Variable(torch.Tensor())
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []


    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1))
        return model(x)


policy = torch.load('D:/Models/VPG.pt')
policy.eval()

def select_action(state):
    # Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    state = torch.from_numpy(state).type(torch.FloatTensor)
    state = policy(torch.autograd.Variable(state))
    c = Categorical(state)
    action = c.sample()

    # Add log probability of our chosen action to our history
    inspect_var = policy.policy_history.dim()
    if len(policy.policy_history) > 0:
        policy.policy_history = torch.cat([policy.policy_history, c.log_prob(action).reshape(1)])
    else:
        policy.policy_history = c.log_prob(action).reshape(1)
    return action

def test(episodes=10):
    for episode in range(episodes):
        state = env.reset()  # Reset environment and record the starting state
        env.render()
        done = False

        for time in range(1000):
            action = select_action(state)
            # Step through environment using chosen action
            state, reward, done, _ = env.step(action.cpu().numpy())
            env.render()

            if done:
                break

test()

