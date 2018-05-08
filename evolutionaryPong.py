#!/usr/bin/env python3
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

from tqdm import tqdm
import random
import math
import operator
import matplotlib.pyplot as plt
import numpy as np
use_cuda = torch.cuda.is_available()

class Net(nn.Module):
	def __init__(self,  num_input_channels, num_output_channels, out2, pool_size, stride, linear_nodes):
		super(Net, self).__init__()
		
		self.num_input_channels = num_input_channels
		self.num_output_channels = num_output_channels
		self.out2 = out2
		self.pool_size = pool_size
		self.stride = stride
		self.linear_nodes = linear_nodes

		#layers
		self.conv1 = nn.Conv2d(num_input_channels, num_output_channels, stride)
		self.pool = nn.MaxPool2d(pool_size, pool_size)
		self.conv2 = nn.Conv2d(num_output_channels, out2, stride)
		self.fc1 = nn.Linear(out2*(stride+2)**2*(stride+num_output_channels), linear_nodes[0])
		self.fc2 = nn.Linear(linear_nodes[0], linear_nodes[1])
		self.fc3 = nn.Linear(linear_nodes[1], linear_nodes[2])


	def forward(self, input):
		res = self.pool(F.relu(self.conv1(input)))
		res = self.pool(F.relu(self.conv2(res)))
		res = res.view(-1, self.out2*(self.stride+2)**2*(self.stride+self.num_output_channels))
		res = F.relu(self.fc1(res))
		res = F.relu(self.fc2(res))
		res = self.fc3(res)
		return res

	def getParams(self):
		return self.num_input_channels, self.num_output_channels, self.out2, self.pool_size, self.stride, self.linear_nodes

#makes a bunch of different nets
def makeNets(numNets):
	nets = []
	print("MAKING NETS")
	for i in tqdm(range(numNets)):
		num_input_channels = 3
		num_output_channels = random.randrange(1, 100)
		out2 = random.randrange(1, 100)
		pool_size = random.randrange(1, 5)
		stride = random.randrange(1,5)
		linear_nodes = [random.randrange(100,300), random.randrange(30, 100), 2]
		net = Net(num_input_channels, num_output_channels, out2, pool_size, stride, linear_nodes)

		nn.init.xavier_uniform(net.conv1.weight)
		nn.init.xavier_uniform(net.conv2.weight)
		nn.init.xavier_uniform(net.fc1.weight)
		nn.init.xavier_uniform(net.fc2.weight)
		nn.init.xavier_uniform(net.fc3.weight)

		nn.init.uniform(net.conv1.bias)
		nn.init.uniform(net.conv2.bias)
		nn.init.uniform(net.fc1.bias)
		nn.init.uniform(net.conv1.bias)
		nn.init.uniform(net.fc2.bias)
		nn.init.uniform(net.fc3.bias)

		net = net.cuda() if use_cuda else net
		nets.append(net)
	
	return nets

#running nets in environment
def inference(nets):
	env = gym.make("Pong-v0")
	num_episodes = 5
	num_iters = 10000
	rewards = np.array([])
	for net in nets:
		total_reward = 0
		observation = env.reset()
		for i_episode in tqdm(range(num_episodes)):
			for _ in range(num_iters):
				res = net(Variable(observation.view(1, 3, 210, 160)))
				action = 2 if res.data[0][0] > res.data[0][1] else 3
				
				observation, reward, done, info = env.step(action)
				total_reward += reward
				if done:
					print("Finished after %d timesteps", _)
					break
				if i_episode == num_iters - 1:
					print("gone through %d iters", num_iters)
			
		np.append(rewards, total_reward)
	return rewards

#Editing Nets based off of results
def evolution(rewards, nets, survival_rate, exploration_rate, combine_rate):
	evolved_nets = []
	numNets = len(nets)

	numSurvivors = math.floor(numNets*survival_rate)
	numRescued = math.floor(numNets*exploration_rate)
	numCombined = math.floor(numNets*combine_rate)
	numMutated = numNets - numSurvivors - numRescued - numCombined

	def naturalSelection():
		index, value = max(enumerate(rewards), key=operator.itemgetter(1))
		evolved_nets.append(nets[index])
		rewards.pop(index)
		nets.pop(index)

	def combine(tensor1, tensor2): #cross products 
		size1 = tensor1.size()
		size2 = tensor2.size()
		tensor1 = tensor1.view(1, -1)
		tensor2 = tensor2.view(1, -1)
		tensor1Len = tensor1.size()[1]
		tensor2Len = tensor2.size()[1]

		if tensor1Len > tensor2Len:
			res = torch.cat(torch.cross(tensor1[:,:tensor2Len + 1], tensor2), tensor1[:,tensor2Len + 1:]).view(size1)
		elif tensor1Len < tensor2Len:
			res = torch.cat(torch.cross(tensor2[:,:tensor1Len + 1], tensor1), tensor2[:,tensor1Len + 1:]).view(size2)
		else:
			res = torch.cross(tensor1, tensor2).view(size1)
		
		return res
	
	def mutate(tensor):
		size = tensor.size()
		tensor = tensor.view(1, -1)
		for element in tensor:
			element = random.random()

		return tensor.view(size)

	#pick survivors
	for i in range(numSurvivors):
		naturalSelection()
	
	#Combine some
	for i in range(numCombined):
		net1 = random.choice(evolved_nets)
		net2 = random.choice(evolved_nets)
		net1Params = net1.getParams()
		net2Params = net2.getParams()
		newNet = Net(3, max(net1Params[1], net2Params[1]), max(net1Params[2], net2Params[2]),  max(net1Params[3], net2Params[3]),  max(net1Params[4], net2Params[4]),  max(net1Params[5], net2Params[5]))
		
		newNet.conv1.weight = combine(net1.conv1.weight, net2.conv1.weight)
		newNet.conv2.weight = combine(net1.conv2.weight, net2.conv2.weight)
		newNet.fc1.weight = combine(net1.fc1.weight, net2.fc1.weight)
		newNet.fc2.weight = combine(net1.fc2.weight, net2.fc2.weight)
		newNet.fc3.weight = combine(net1.fc3.weight, net2.fc3.combine)
		
		newNet.conv1.bias = combine(net1.conv1.bias, net2.conv1.bias)
		newNet.conv2.bias = combine(net1.conv2.bias, net2.conv2.bias)
		newNet.fc1.bias = combine(net1.fc1.bias, net2.fc1.bias)
		newNet.fc2.bias = combine(net1.fc2.bias, net2.fc2.bias)
		newNet.fc3.bias = combine(net1.fc3.bias, net2.fc3.bias)
		
		newNet = newNet.cuda() if use_cuda else newNet
		evolved_nets.append(newNet)
	
	#pick Rescued
	for i in range(numRescued):
		rescuee = random.choice(nets)
		idx = nets.index(rescuee)
		evolved_nets.append(rescuee)
		nets.pop(rescuee) 

	#mutate Some
	for i in range(numMutated):
		chosenNet = random.choice(nets)
		idx = nets.index(chosenNet)

		chosenNet.conv2.weight = mutate(chosenNet.conv2.weight)
		chosenNet.fc1.weight = mutate(chosenNet.fc1.weight)
		chosenNet.fc2.weight = mutate(chosenNet.fc2.weight)
		chosenNet.fc3.weight = mutate(chosenNet.fc3.weight)
		
		chosenNet.conv1.bias = mutate(chosenNet.conv1.bias)
		chosenNet.conv2.bias = mutate(chosenNet.conv2.bias)
		chosenNet.fc1.bias = mutate(chosenNet.fc1.bias)
		chosenNet.fc2.bias = mutate(chosenNet.fc2.bias)
		chosenNet.fc3.bias = mutate(chosenNet.fc3.bias)

		evolved_nets.append(chosenNet)
		nets.pop(idx)

	return evolved_nets
		
	

#TRAINING
survival_rate = 0.4
exploration_rate = 0.3
combine_rate = 0.2
numEvolutions = 10000
numNets = 10000

def train(survival_rate, exploration_rate,  combine_rate, numEvolutions=10000, numNets=10000):
	avgRewards = np.array([])	
	nets = makeNets(numNets)
	print("TRAINING")

	#analyzation
	def stats(rewards, iteration, print_every=500):
		index, value = max(enumerate(rewards), key=operator.itemgetter(1))
		avg_reward = sum(rewards)/float(len(rewards))
		np.append(avgRewards, avg_reward)
		if iteration % print_every == 0:
			print("Average Reward: %f" % avg_reward)
			print("Best Net: Net %d\n Score: %f" % (index, value))
			iterations = np.array([i for i in range(iteration)])
			fig, ax = plt.subplots()
			fit = np.polyfit(iterations, avgRewards, deg=1)
			ax.plot(x, fit[0] * x + fit[1], color='red')
			print("Change in Average Reward per Iteration: %d" % fit[0])
			ax.scatter(iterations, avgRewards)
			fig.show()
			plt.savefig('plt.png')


	# EVOLVING
	for n_iter in tqdm(range(numEvolutions)):
		print("EVOLVING")
		rewards = inference(nets)
		nets = evolution(rewards, nets, survival_rate, exploration_rate, combine_rate)
		stats(rewards, n_iter)
		exploration_rate = 0.3 - n_iter/6000 
		combine_rate = 0.2 + n_iter/9000
	
	totalRewards = np.zeros(numNets)
	for n_iter in tqdm(range(numEvolutions/10)):
		print("TESTING")
		rewards = inference(nets)
		totalRewards += rewards
	
	totalRewards /= numEvolutions/10
	index, value = max(enumerate(totalRewards), key=operator.itemgetter(1))
	bestNet = nets[index]

	return bestNet

bestNet = train(survival_rate, exploration_rate, combine_rate)
torch.save(bestNet, 'Pongexpert.pt')

def play(net):
	env = gym.make("Pong-v0")
	num_episodes = 5
	num_iters = 10000
	observation = env.reset()
	total_reward = 0
	for i_episode in tqdm(range(num_episodes)):
			for _ in range(num_iters):
				res = net(Variable(observation.view(1, 3, 210, 160)))
				action = 2 if res.data[0][0] > res.data[0][1] else 3
				
				observation, reward, done, info = env.step(action)
				total_reward += reward
				if done:
					print("Finished after %d timesteps", _)
					break
				if i_episode == num_iters - 1:
					print("gone through %d iters", num_iters)
	
	return total_reward
