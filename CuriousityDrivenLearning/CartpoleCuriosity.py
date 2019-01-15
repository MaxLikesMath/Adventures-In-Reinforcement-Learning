import torch
import torch.nn as nn
import gym
import numpy as np
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 0
torch.manual_seed(SEED)
env = gym.make('CartPole-v1').unwrapped
env.seed(1)

action_size = env.action_space.n
state_size = env.observation_space.shape[0]

encoded_size = 4
gamma = 0.95
beta = 0.2


def plot(rewards_plot):
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('Reward Over Time')
    plt.plot(rewards_plot)
    plt.show()

class Exploration_Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super(Exploration_Policy, self).__init__()
        self.action_size = action_size
        self.state_size = state_size

        self.layer1 = nn.Linear(state_size, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, action_size)

    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = F.sigmoid(self.layer3(out))
        return out


class Forward(nn.Module):
    def __init__(self, encoded_size, action_size):
        super(Forward, self).__init__()
        self.action_size = action_size
        self.encoded_size = encoded_size

        self.layer1 = nn.Linear(encoded_size + action_size, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, encoded_size)

    def forward(self, x, x1):
        x = torch.cat([x, x1], -1)
        out =F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = self.layer3(out)
        return out


class Inverse(nn.Module):
    def __init__(self, encoded_size, action_size):
        super(Inverse, self).__init__()
        self.action_size = action_size
        self.encoded_size = encoded_size

        self.layer1 = nn.Linear(encoded_size * 2, 8)
        self.layer2 = nn.Linear(8, 8)
        self.layer3 = nn.Linear(8, 8)
        self.layer4 = nn.Linear(8, 8)
        self.layer5 = nn.Linear(8, action_size)

    def forward(self, x, x1):
        x = torch.cat([x, x1], -1)
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = F.relu(self.layer3(out))
        out = F.relu(self.layer4(out))
        out = (self.layer5(out))

        return out


class Encoder(nn.Module):
    def __init__(self, encoded_size, state_size):
        super(Encoder, self).__init__()
        self.action_size = action_size
        self.encoded_size = encoded_size

        self.layer1 = nn.Linear(state_size, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 4)
        self.layer4 = nn.Linear(4, encoded_size)

    def forward(self, x):
        out = F.elu(self.layer1(x))
        out = F.elu(self.layer2(out))
        out = F.elu(self.layer3(out))
        out = self.layer4(out)
        return out


enc_net = Encoder(encoded_size, state_size)
for_net = Forward(encoded_size, action_size)
inv_net = Inverse(encoded_size, action_size)
explore_agent = Exploration_Policy(state_size, action_size)

icm_optimizer = torch.optim.Adam(list(enc_net.parameters()) +
                                 list(for_net.parameters()) +
                                 list(inv_net.parameters()), lr=0.001)


agent_optimizer = torch.optim.Adam(explore_agent.parameters(), lr=0.01)

batch_size = 5
average_reward = []
losses = []
for e in range(1000):
    state = env.reset()
    state = torch.from_numpy(state).type("torch.FloatTensor")
    all_rewards = []
    all_actions = []
    all_probs = []
    all_states = []
    all_new_states = []
    all_for_loss = []
    total_reward = 0
    steps = 0
    first = True
    while True:
        all_states.append(state)
        steps += 1
        agent_action = explore_agent(state)
        action_distribution = torch.softmax(agent_action, -1)

        m = Categorical(action_distribution)
        action = m.sample()

        all_actions.append(action)
        all_probs.append(action_distribution[action])

        new_state, reward, done, info = env.step(action.item())
        total_reward += reward
        all_rewards.append(reward)

        new_state = torch.from_numpy(new_state).type("torch.FloatTensor")
        all_new_states.append(new_state)
        if ((steps - 1) % batch_size == 0 or done) and (steps - 1) != 0:

            all_states = torch.stack(all_states).type("torch.FloatTensor")
            all_new_states = torch.stack(all_new_states).type("torch.FloatTensor")
            all_actions = torch.stack(all_actions)
            if True:
                enc_state = enc_net(all_states)
                enc_new_state = enc_net(all_new_states)

                action_o = torch.zeros([all_states.size(0), action_distribution.size(0)])
                action_o = action_o.scatter(1, all_actions.unsqueeze(-1), 1)

                inv_out = inv_net(enc_state, enc_new_state)
                inv_out = F.softmax(inv_out)
                icm_optimizer.zero_grad()
                inv_loss = torch.mean(nn.CrossEntropyLoss()(inv_out, all_actions)) * (1 - beta)

                for_out = for_net(action_o, enc_state)

                for_loss = torch.mean((torch.pow(for_out - enc_new_state, 2)), -1)
                for_loss_mean = torch.mean(for_loss) * beta


                icm_loss = inv_loss + for_loss_mean
                icm_loss.backward(retain_graph=True)
                icm_optimizer.step()

                all_for_loss.append(for_loss)
                all_actions = []
                all_states = []
                all_new_states = []
        first = False
        if done:
            average_reward.append(total_reward)
            losses.append(inv_loss.item())
            all_probs = torch.stack(all_probs)
            all_for_loss = torch.cat(all_for_loss, 0)


            all_rewards = torch.from_numpy(np.array(all_rewards)).type("torch.FloatTensor")
            all_rewards += all_for_loss

            running_add = 0
            for i in reversed(range(all_rewards.size(0))):
                running_add = running_add * gamma + all_rewards[i]
                all_rewards[i] = running_add

            all_rewards = (all_rewards - torch.mean(all_rewards)) / (torch.std(all_rewards) + 0.0000001)

            agent_optimizer.zero_grad()

            loss = torch.mean(-torch.log(all_probs) * all_rewards)
            loss.backward()
            agent_optimizer.step()

            all_rewards = []
            all_probs = []

            print("Episode : {}  Rewards : {}".format(e + 1, np.sum(total_reward)))
            break

        state = new_state
plot(average_reward)
plot(losses)
print(np.mean(average_reward))
torch.save(enc_net, 'D:/Models/Enc.pt')
torch.save(for_net, 'D:/Models/For.pt')
torch.save(explore_agent, 'D:/Models/Age.pt')
torch.save(inv_net, 'D:/Models/Inv.pt')