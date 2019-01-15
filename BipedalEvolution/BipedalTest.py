import numpy as np
from TimedAgent import Agent
from BiPedalModel import Model
model=Model()




agent=Agent(model, POPULATION_SIZE=100, LEARNING_RATE=0.01, EXPLORATION_DEC_STEPS = 1500000, num_thread = 1,)
agent.load('BiModel.pkl')
agent.play(10, render=True)