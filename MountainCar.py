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

### Set parameters ###
num_episodes = 1
num_individuals = 20
dimensions = [2, 1]
activation_hidden = 'relu'
activation_output = 'sigmoid'
num_timesteps = 400
num_generations = 5
######################

gen = Generation(num_individuals = num_individuals, dimensions = dimensions, activation_hidden = activation_hidden, activation_output = activation_output)
#print(gen)

avg_fitness_history = []

for gen_num in range(num_generations):

	print('Generation ', gen_num)
	# Loop over the individuals in the population in one generation
	for individual in gen.get_population():

		observation = env.reset()

		fitness = 0
		# Loop over the number of time steps
		for t in range(num_timesteps):
			env.render()
			observation = np.reshape(observation, [1, 4])
			#print('Observation : ', type(observation), observation, observation.shape)
			action = individual.predict(observation)
			#print('action = ', action)
			observation, reward, done, info = env.step(action)
			#print('-----------------------------')
			
			#time.sleep(0.1)
			if done:
				break

			fitness += 1

		print('Episode finished after {} timesteps'.format(fitness+1))
		individual.set_fitness(fitness+1)

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

