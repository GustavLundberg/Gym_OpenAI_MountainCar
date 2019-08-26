import time
import gym
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from Generation import *
from NeuralNetwork import *

env = gym.make('MountainCar-v0')

#model = NeuralNetwork(dimensions = [5, 3, 1], activation_hidden = 'relu')
#print('weights : ', model.getModel().get_weights())

### Environment 'settings' ###
num_actions = 3
num_inputs = 2
############################## 

##########################################
############# Set parameters #############
##########################################

num_episodes = 1
num_individuals = 30
dimensions = [2, 3]
activation_hidden = 'relu'
activation_output = 'softmax'
num_timesteps = 120
num_generations = 10

pos_weight = 1
speed_weight = 10

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

		observation = env.reset()

		fitness = 0
		best_position = -2 # Best position during the entire episode
		max_speed = 0 # Note that this is the magnitude if the velocity, i.e. direction does not matter
		# Loop over the number of time steps
		for t in range(num_timesteps):
			env.render()
			observation = np.reshape(observation, [1, num_inputs])
			#print('Observation : ', type(observation), observation, observation.shape)
			
			# For multiclass classification, choose predict_multiclass
			# For binary classification, choose predict_binary
			action = individual.predict_multiclass(observation)
			
			#print('action = ', action)
			observation, reward, done, info = env.step(action)
			#print('-----------------------------')
			if observation[0] > best_position:
				best_position = observation[0]

			if abs(observation[1]) > max_speed:
				max_speed = observation[1]

			# fitness += reward
			#time.sleep(0.1)
			if done:
				break

		fitness = pos_weight * best_position + speed_weight * max_speed # Using best_position and max_speed as fitness instead of using the reward of the environment
		print('Fitness for this episode = {}'.format(fitness))
		individual.set_fitness(fitness)

	avg_fitness_history.append(mean(gen.get_fitness()))

	# Print weights
	#for ind in gen.get_population():
	#	print('Before breed: weights = ', ind.get_weights())
	#	print('--------------------------------------------')

	#parents = gen.select_parents()
	gen.breed_population()

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

