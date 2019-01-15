# Adventures-In-Reinforcement-Learning
This repository contains code for different implementations of assorted reinforcement learning algorithms (evolutionary ones as well). My aim is to create a place where algorithms can be explored and manipulated easily, and implemented on interesting environments.

## Basic Documentation
This repository uses a multitude of common Python libraries, such as matplotlib and NumPy.
For most of the actual machine learning done in here, PyTorch is the weapon of choice, the oonly exception being
in the BiPedal Evolution, which uses the EvoStra library: https://github.com/alirezamika/evostra. The environments are
all available in OpenAI-Gym, however, I use the PyBullet implementations of the MuJoCo environments, available here: https://github.com/benelot/pybullet-gym

## Basic Project Descriptions

### BiPedal Evolution
This project follows an example that was posted by the creator of EvoStra, however, their training parameters didn't work. So, I made this
so people could train the agent from scratch to see how it works. The package is super fun and easy to use, and I highly suggest
checking it out!

### Curiousity Driven Learning
This project currently contains only one simple implementation of the paper *Curiosity-driven Exploration by Self-supervised Prediction*
which can be found here: https://arxiv.org/abs/1705.05363 which I would like to apply to a more complex environment. 
However, due to computational restrictions I've not had the time to really try. 

### Half-Cheetah Using Proximal Policy Optimization
A long time ago, I tried implementing PPO in Tensorflow to solve the Lunar Lander environment. It didn't work great, but I learned
a lot doing it. Recently I have been experimenting with PPO for the humanoid walker environment, which hopefully
I will remember to finish and upload. However, the first thing I did was write the algorithm to solve the easier
Half-Cheetah Environment. The current parameter settings are good to get about a score of 1000, but could certainly be improved 
upon with relative ease.

### Lunar Lander Using Vanilla Policy Gradients
So, as I said, this environment defeated me in my ML youth. Recently, I'd been working on a research project that involved using 
policy gradient methods and decided it was time to extract my revenge on the Lunar Lander. A quick bit of PyTorch later, and I had 
avenged myself. Vanilla Policy Gradient methods tend to be rather unstable, so in more complex environments something like
TRPO, PPO, or A2C should be used but I think it's a fantastically beautiful piece of math. I think anyone with any ML interest should
know and understand policy gradients.


