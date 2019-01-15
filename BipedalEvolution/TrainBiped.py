import numpy as np
from TimedAgent import Agent
from BiPedalModel import Model
model=Model()




agent=Agent(model, POPULATION_SIZE=25, LEARNING_RATE=0.03, EXPLORATION_DEC_STEPS = 1000000, num_thread = 1,  SIGMA = 0.1, LR_mode = 0)
agent.train(600)
agent.save('BiModel.pkl')
agent.play(10, render=True)