import numpy as np
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def get_action(mu, std):
    action = torch.normal(mu, std)
    action = action.data.numpy()
    return action

def plot(rewards_plot):
    plt.figure(figsize=(10, 5))
    plt.subplot(131)
    plt.title('Reward Over Time')
    plt.plot(rewards_plot)
    plt.show()


class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        super(Actor, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(num_inputs, 128),nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(128, 64),nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
        self.fc4 = nn.Linear(32, num_outputs)
        self.fc4.weight.data.mul_(0.1)
        self.fc4.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        mu = self.fc4(x)
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)
        return mu, std, logstd


class Critic(nn.Module):
    def __init__(self, num_inputs):
        super(Critic, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(num_inputs, 128),nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(128, 64),nn.ReLU())
        self.fc3 = nn.Linear(64, 1)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        value = self.fc3(x)
        return value


def log_density(x, mu, std, logstd):
    var = std.pow(2)
    log_density = -(x - mu).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - logstd
    return log_density.sum(1, keepdim=True)


class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape

    def save(self, env_name):
        saveable_M = self._M
        saveable_S = self._S
        saveable_n = self._n
        np.save('D:/Numpy Arrays/{}_State_Mean.npy'.format(env_name), saveable_M)
        np.save( 'D:/Numpy Arrays/{}_State_Deviation.npy'.format(env_name), saveable_S)
        np.save( 'D:/Numpy Arrays/{}_State_n.npy'.format(env_name), [saveable_n])

    def load(self, env_name):
        self._M = np.load('D:/Numpy Arrays/{}_State_Mean.npy'.format(env_name))
        self._S = np.load('D:/Numpy Arrays/{}_State_Deviation.npy'.format(env_name))
        self._n = np.load('D:/Numpy Arrays/{}_State_n.npy'.format(env_name))[0]


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape

    def load(self, env_name):
        self.rs.load(env_name)

    def save(self, env_name):
        self.rs.save(env_name)








def get_gae(rewards, masks, values, gamma = 0.995, lam = 0.95):
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)

    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + gamma * running_returns * masks[t]
        running_tderror = rewards[t] + gamma * previous_value * masks[t] - values.data[t]
        running_advants = running_tderror + gamma * lam * running_advants * masks[t]

        returns[t] = running_returns
        previous_value = values.data[t]
        advants[t] = running_advants

    advants = (advants - advants.mean()) / advants.std()
    return returns, advants


def surrogate_loss(actor, advants, states, old_policy, actions, index):
    mu, std, logstd = actor(torch.Tensor(states))
    new_policy = log_density(actions, mu, std, logstd)
    old_policy = old_policy[index]

    ratio = torch.exp(new_policy - old_policy)
    surrogate = ratio * advants
    return surrogate, ratio

def train_model(actor, critic, memory, actor_optim, critic_optim, batch_size=32, clip_param=0.2):
    '''We first access the data in memory, creating a list of old states by vertically stacking the memory, passing
    it as a Torch tensor to our Critic network to predict values. We then extract our previous actions, rewards, and masks from memory.
    '''
    memory = np.array(memory)
    states = np.vstack(memory[:, 0])
    values = critic(torch.Tensor(states))
    actions = list(memory[:, 1])
    rewards = list(memory[:, 2])
    masks = list(memory[:, 3])


    '''
    Now, to optimize our policy, we compute our advantage score using our function as described earlier. We also get the
    distribution as modeled by our actor. Finally, we get our old policy distribution by calculating the log density
    of our previous actions.
    '''
    returns, advants = get_gae(rewards, masks, values)
    mu, std, logstd = actor(torch.Tensor(states))
    old_policy = log_density(torch.Tensor(actions), mu, std, logstd)

    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)


    for epoch in range(10):
        np.random.shuffle(arr)

        '''
        First, we split our current set of experience data into batches, unsqueezing them into tensors of the proper dimension.
        Next, we compute our loss using our surrogate loss function which passes us back the new policy/old policy ratio
        multiplied by the advantage for each sample in our batch. We then train our critic by getting its values for the input
        then calculating the MSE between its estimates, and the actual return. next we define our actor loss as the minimum
        between either the original surrogate loss or the clipped version of the surrogate loss (this is what makes PPO so powerful)
        We then update our actor. 
        '''

        for i in range(n // batch_size):
            batch_index = arr[batch_size * i: batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)
            inputs = torch.Tensor(states)[batch_index]
            returns_samples = returns.unsqueeze(1)[batch_index]
            advants_samples = advants.unsqueeze(1)[batch_index]
            actions_samples = torch.Tensor(actions)[batch_index]

            loss, ratio = surrogate_loss(actor, advants_samples, inputs, old_policy.detach(), actions_samples, batch_index)

            values = critic(inputs)
            critic_loss = criterion(values, returns_samples)
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            clipped_ratio = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
            clipped_loss = clipped_ratio * advants_samples
            actor_loss = -torch.min(loss, clipped_loss).mean()

            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()