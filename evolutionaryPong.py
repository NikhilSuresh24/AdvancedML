#!/usr/bin/env python3
import gym
import torch
import torch.nn as nn
import torch.nn.Functional as F
from torch.autograd import Variable

from tqdm import tqdm
import random
use_cuda = torch.cuda.is_available()

env = gym.make("Pong-v0")
num_episodes = 5
total_reward = 0
num_iters = 1000
# 2 is up
# 3 is down
for i_episode in tqdm(range(num_episodes)):
	env.reset()
	print("i_episode: ", i_episode)
	for _ in range(num_iters):
		action = 2 if random.random() < 0.5 else 3
		observation, reward, done, info = env.step(action)
		total_reward += reward
		print(total_reward)
		env.render()
		if done:
			print("Finished after %d timesteps", _)
			break
		if _ == num_iters - 1:
			print("gone through %d iters", num_iters)

env.close()

class Net(nn.module):
	def __init__(self, input_size, output_size, num_input_channels):
		super("Net", self).__init__()
		self.input_size = input_size
		self.output_size = output_size

		self.conv1 = nn.Conv2d(num_input_channels, 1)
		self.out = nn.Linear()


	def forward(self, input, hidden):

	


#If you need to apply the initialisation to a specific module,
#say conv1, you can extract the specific parameters with conv1Params = list(net.conv1.parameters()). 
#You will have the kernels in conv1Params[0] and the bias terms in conv1Params[1].
# extract weights from pytorch layer using
# model.layer[0].weight

#This site
#https://discuss.pytorch.org/t/weight-initilzation/157/7
#include mutations


