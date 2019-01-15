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
env.seed(seed)
torch.manual_seed(seed)

learning_rate = 0.01
gamma = 0.99


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


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)


def select_action(state):
    # Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    state = torch.from_numpy(state).type(torch.FloatTensor)
    state = policy(torch.autograd.Variable(state))
    c = Categorical(state)
    action = c.sample()

    # Add log probability of our chosen action to our history
    if len(policy.policy_history) > 0:
        policy.policy_history = torch.cat([policy.policy_history, c.log_prob(action).reshape(1)])
    else:
        policy.policy_history = c.log_prob(action).reshape(1)
    return action

def update_policy():
    R = 0
    rewards = []

    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0, R)

    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    # Calculate loss
    loss = (torch.sum(torch.mul(policy.policy_history, torch.autograd.Variable(rewards)).mul(-1), -1))

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save and intialize episode history counters
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = torch.autograd.Variable(torch.Tensor())
    policy.reward_episode = []


def main(episodes):
    running_reward = 0
    for episode in range(episodes):
        state = env.reset()  # Reset environment and record the starting state
        done = False
        episode_reward = 0
        while not done:
            action = select_action(state)
            # Step through environment using chosen action
            state, reward, done, _ = env.step(action.cpu().numpy())
            episode_reward += reward
            # Save reward
            policy.reward_episode.append(reward)

        # Used to determine when the environment is solved.
        #running_reward = (running_reward * 0.99) + (time * 0.01)
        running_reward += episode_reward
        update_policy()

        if episode % 10 == 0:
            print('Episode {}\tLast Reward: {:.2f}\tAverage Reward: {:.2f}'.format(episode, episode_reward, running_reward/episode))

def test(episodes=10):
    for episode in range(episodes):
        state = env.reset()  # Reset environment and record the starting state
        env.render()
        done = False

        while not done:
            action = select_action(state)
            # Step through environment using chosen action
            state, reward, done, _ = env.step(action.cpu().numpy())
            env.render()

            if done:
                break

episodes = 1000
main(episodes)
torch.save(policy, 'D:/Models/VPG.pt')
test()




