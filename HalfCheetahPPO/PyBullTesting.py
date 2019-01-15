
import numpy as np
#import gym
import gym.envs
import torch
from HalfCheetah.PyBullUtils import ZFilter, get_action
import pybullet as p

seed=123
env = gym.make("HalfCheetahPyBulletEnv-v0")
env.seed(seed)
torch.manual_seed(seed)
env.render()
env.reset()

for i in range(p.getNumBodies()):
    print(p.getBodyInfo(i))
    if p.getBodyInfo(i)[1].decode() == "cheetah":
        torsoId = i





num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

running_state = ZFilter((num_inputs,), clip=5)
running_state.load('HC')



actor = torch.load('D:/Models/HCActor1.pt')
critic = torch.load('D:/Models/HCCritic1.pt')

def test(iterations = 10):
    actor.eval(), critic.eval()
    scores = []
    distance = 5
    yaw = 0
    for iter in range(iterations):
        steps = 0
        rewards = []
        state = env.reset()
        env.render()
        state = running_state(state)
        score = 0
        done = False
        while not done:
            mu, std, _ = actor(torch.Tensor(state).unsqueeze(0))
            action = get_action(mu, std)[0]
            next_state, reward, done, _ = env.step(action)
            next_state = running_state(next_state)
            rewards.append(reward)
            score += reward
            steps +=1
            state = next_state
            humanPos = p.getLinkState(torsoId,4)[0]
            p.resetDebugVisualizerCamera(distance,yaw,-20,humanPos)
            env.render()
        print('Episode {} Finished After {} Steps With Score {}'.format(iter, steps, score))
        scores.append(score)
    print('Testing Completed With Average Score {}'.format(np.mean(scores)))


test()






