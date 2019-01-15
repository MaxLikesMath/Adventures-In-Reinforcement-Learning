import random
import pickle
import numpy as np
from evostra import EvolutionStrategy
from BiPedalModel import Model
import gym
import time



class Agent:

    def __init__(self, model, training_steps=500, environment='BipedalWalker-v2', AGENT_HISTORY_LENGTH = 1, POPULATION_SIZE = 50,
    EPS_AVG = 1, SIGMA = 0.1, LEARNING_RATE = 0.01, INITIAL_EXPLORATION = 1.0, FINAL_EXPLORATION = 0.0,
    EXPLORATION_DEC_STEPS = 10000, num_thread=1, LR_mode = 0):
        self.env = gym.make(environment)
        self.model = model
        self.exploration = INITIAL_EXPLORATION
        self.training_steps = training_steps
        self.AGENT_HISTORY_LENGTH = AGENT_HISTORY_LENGTH
        self.POPULATION_SIZE = POPULATION_SIZE
        self.EPS_AVG = EPS_AVG
        self.SIGMA = SIGMA
        self.LEARNING_RATE = LEARNING_RATE
        self.INITIAL_EXPLORATION = INITIAL_EXPLORATION
        self.FINAL_EXPLORATION = FINAL_EXPLORATION
        self.EXPLORATION_DEC_STEPS = EXPLORATION_DEC_STEPS
        self.num_thread = num_thread
        self.LR_mode = LR_mode
        self.es = EvolutionStrategy(self.model.get_weights(), self.get_reward, self.POPULATION_SIZE, self.SIGMA,
                                    self.LEARNING_RATE, num_threads=num_thread, LR_mode=self.LR_mode)

    def get_predicted_action(self, sequence):
        prediction = self.model.predict(np.array(sequence))
        return prediction

    def load(self, model_file):
        with open(model_file, 'rb') as fp:
            self.model.set_weights(pickle.load(fp))
        self.es.weights = self.model.get_weights()

    def save(self, model_file):
        with open(model_file, 'wb') as fp:
            pickle.dump(self.es.get_weights(), fp)

    def train(self, iterations):
        print('Training')
        self.es.run(iterations, print_step=1)
        optimized_weights = self.es.get_weights()
        self.model.set_weights(optimized_weights)

    def play(self, episodes, render=True):
        self.model.set_weights(self.es.weights)
        for episode in range(episodes):
            print('On episode number {}'.format(episode))
            total_reward = 0
            observation = self.env.reset()
            sequence = [observation] * self.AGENT_HISTORY_LENGTH
            done = False
            while not done:
                if render:
                    self.env.render()
                action = self.get_predicted_action(sequence)
                observation, reward, done, _ = self.env.step(action)
                total_reward += reward
                sequence = sequence[1:]
                sequence.append(observation)
            print("total reward:", total_reward)

    def get_reward(self, weights):
        total_reward = 0.0
        self.model.set_weights(weights)

        for episode in range(self.EPS_AVG):
            start_time = time.time()
            observation = self.env.reset()
            sequence = [observation] * self.AGENT_HISTORY_LENGTH
            done = False
            while not done:
                self.exploration = max(self.FINAL_EXPLORATION, self.exploration - self.INITIAL_EXPLORATION / self.EXPLORATION_DEC_STEPS)
                if random.random() < self.exploration:
                    action = self.env.action_space.sample()
                else:
                    action = self.get_predicted_action(sequence)
                observation, reward, done, _ = self.env.step(action)
                total_reward += reward
                sequence = sequence[1:]
                sequence.append(observation)
        #print("total reward: ", total_reward)
        #print('Finished in {} seconds'.format(time.time() - start_time))
        return total_reward / self.EPS_AVG

