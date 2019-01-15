import gym.envs
import torch.optim as optim
from collections import deque
from HalfCheetah.PyBullUtils import *

seed=123
env = gym.make("HalfCheetahPyBulletEnv-v0")
env.seed(seed)
torch.manual_seed(seed)


num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

print('state size:', num_inputs)
print('action size:', num_actions)

actor = Actor(num_inputs, num_actions)
critic = Critic(num_inputs)

#actor = torch.load('D:/Models/HCActor1.pt')
#critic = torch.load('D:/Models/HCCritic1.pt')

actor_optim = optim.Adam(actor.parameters(), lr=0.0001)
critic_optim = optim.Adam(critic.parameters(), lr=0.0001)

running_state = ZFilter((num_inputs,), clip=5)
#running_state.load('HC')

def train():
    save_score = 400
    av_score_over_time = []
    episodes = 0
    for iter in range(15000):
        if episodes>=10000:
            break
        actor.eval(), critic.eval()
        memory = deque()

        steps = 0
        scores = []
        while steps < 5000:
            episodes += 1
            state = env.reset()
            state = running_state(state)
            score = 0
            for _ in range(10000):
                steps += 1
                mu, std, _ = actor(torch.Tensor(state).unsqueeze(0))
                action = get_action(mu, std)[0]
                next_state, reward, done, _ = env.step(action)
                next_state = running_state(next_state)

                if done:
                    mask = 0
                else:
                    mask = 1

                memory.append([state, action, reward, mask])

                score += reward
                state = next_state

                if done:
                    break
            scores.append(score)
        score_avg = np.mean(scores)
        av_score_over_time.append(score_avg)
        print('{} episode score is {:.2f}'.format(episodes, score_avg))
        actor.train(), critic.train()
        train_model(actor, critic, memory, actor_optim, critic_optim, batch_size=128)
        if score_avg > save_score:
            torch.save(actor, 'D:/Models/HCActor1.pt')
            torch.save(critic, 'D:/Models/HCCritic1.pt')
            running_state.save('HC')
            save_score += 1500
    torch.save(actor, 'D:/Models/HCActor1.pt')
    torch.save(critic, 'D:/Models/HCCritic1.pt')
    running_state.save('HC')
    return av_score_over_time

av_score = train()
plot(av_score)