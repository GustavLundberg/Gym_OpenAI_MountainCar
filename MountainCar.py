import time
import gym
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from Generation import *
from NeuralNetwork import *

env = gym.make('MountainCar-v0')

################# NOTES ##################
# It is crucial to normalize the inputs!

### Environment 'settings' ###
num_actions = 3
num_inputs = 2
############################## 

##########################################
############# Set parameters #############
##########################################

num_episodes = 1
num_individuals = 15
dimensions = [num_actions] # For actions > 2, use num_actions number of output neurons, as well as predict_multiclass below
# dimensions = [2, 1] # For num_actions = 2, use 1 neuron in the output layer, as well as predict_binary below
activation_hidden = 'relu'
activation_output = 'softmax'
num_timesteps = 100 # 200 seems to be maximum
num_generations = 7
num_mutations = 2

pos_weight = 1
speed_weight = 1
ratio_mean = 0.1

##########################################
##########################################
##########################################

gen = Generation(num_individuals = num_individuals, dimensions = dimensions, 
	num_inputs = num_inputs, activation_hidden = activation_hidden, activation_output = activation_output)
print(gen)

avg_fitness_history = []

for gen_num in range(num_generations):

	print('Generation ', gen_num)
	# Loop over the individuals in the population in one generation
	for individual in gen.get_population():

		#print('weights = ', individual.get_weights())

		observation = env.reset()

		fitness = 0
		best_position = -2 # Best position during the entire episode
		max_speed = 0 # Note that this is the magnitude if the velocity, i.e. direction does not matter
		#worst_pos = 2
		# Loop over the number of time steps
		for t in range(num_timesteps):
			env.render()
			observation = np.reshape(observation, [1, num_inputs])
			observation[0, 0] = observation[0, 0] + 0.53 # Attempt at 'normalizing' the inputs (Want to make 0 at the point where the slope is zero, seems like 0.53 does the trick)
			observation[0, 1] = 100 * observation[0, 1] # Attempt at 'normalizing' the inputs
			#print(observation)
			
			# For multiclass classification, choose predict_multiclass
			# For binary classification, choose predict_binary
			action = individual.predict_multiclass(observation)
			#print('action = ', action)
			
			#print('action = ', action)
			observation, reward, done, info = env.step(action)
			#print('-----------------------------')
			if observation[0] > best_position:
				best_position = observation[0]

			#if observation[0] < worst_pos:
			#	worst_pos = observation[0]

			if abs(observation[1]) > max_speed:
				max_speed = abs(observation[1])

			#print('max_speed = ', max_speed)

			# The task is completed before time runs out
			if done:
				timesteps_before_completion = t
				#print('timesteps_before_completion = ', timesteps_before_completion)
				break

		#print('best_position = ', best_position)
		#print('worst_pos = ', worst_pos)

		fitness = pos_weight * best_position + speed_weight * max_speed # Using best_position and max_speed as fitness instead of using the reward of the environment
		
		# Attempt at promoting individuals that completes the task. Gives an additional fitness based on the time it took to complete the task
		if done:
			fitness = fitness * (1 + timesteps_before_completion / num_timesteps)

		print('Fitness for this episode = {}'.format(fitness))
		individual.set_fitness(fitness)

	avg_fitness_history.append(mean(gen.get_fitness()))

	# Print weights
	#for ind in gen.get_population():
	#	print('Before breed: weights = ', ind.get_weights())
	#	print('--------------------------------------------')

	#parents = gen.select_parents()
	gen.breed_population(num_mutations = num_mutations, ratio_mean = ratio_mean)

	# Print weights
	#for ind in gen.get_population():
	#	flattened = ind.flatten()
	#	print('Flattened = ', flattened)
		#print('After breed : weights = ', ind.get_weights())
		#print('--------------------------------------------')


env.close()
	
plt.figure(0)
plt.xlabel('Generation')
plt.ylabel('Average fitness')
plt.title('Average fitness over time')
plt.plot(list(range(num_generations)), avg_fitness_history)
plt.show()

