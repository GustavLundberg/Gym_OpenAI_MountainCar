from numpy.random import choice
from statistics import mean
from NeuralNetwork import *

class Generation:

	def __init__(self, num_individuals, dimensions, num_inputs, activation_hidden, activation_output = 'sigmoid'):
		self.num_individuals = num_individuals
		self.dimensions = dimensions
		self.population = [NeuralNetwork(dimensions = dimensions, num_inputs = num_inputs, activation_hidden = activation_hidden, 
			activation_output = activation_output) for _ in range(num_individuals)]

	def __repr__(self):
		return 'Generation object with ' + str(self.num_individuals) + ' individuals'

	def get_population(self):
		return self.population

	def get_fitness(self):
		return [individual.get_fitness() for individual in self.population]

	def sort_population(self):
		
		def fitness_key(neuralNetwork):
			return neuralNetwork.get_fitness()

		self.population = sorted(self.population, key = fitness_key, reverse = True)
		return 

	#def print_population(self):
	#	for i, individual in enumerate(self.population):
			#individual.set_fitness(i)
			#print('weights = ', individual.getModel().get_weights())
			#print('fitness = ', individual.get_fitness())
			#print('----------------------------------------------------')

	# Select parents to be used for breeding
	def select_parents(self):
		self.sort_population()
		fitness = self.get_fitness()
		max_fitness = max(fitness)
		min_fitness = min(fitness)
		if max_fitness != min_fitness:
			probs = [(f-min_fitness)/(max_fitness-min_fitness) for f in fitness] # Normalizing so that all probs is in range [0, 1]
		else:
			probs = [f/max_fitness for f in fitness] # Normalizing so that all probs is in range [0, 1]
		
		### Introduce transforms to the weighted probabilities here ###
		probs = [p if p > mean(probs) else 0 for p in probs]
		###############################################################

		# Normalizing so all probabilites in probs sum to 1
		probs_sum = sum(probs)
		probs = [p/probs_sum for p in probs]

		#print('probabilities = ', probs)

		# Sample from list with weighted probabilities probs
		rand_ints = choice(list(range(self.num_individuals)), self.num_individuals * 2, p = probs)
		return [self.population[i] for i in rand_ints]

	def breed_population(self, num_mutations = 1, ratio_mean = 0.5):
		parents = self.select_parents()

		children = []
		children.append(self.population[0]) # Make sure to keep the best individual from prev generation
		loop_length = len(parents) - 2 		# Use - 2 to compensate for the fact that we are keeping the best individual from prev generation 
		for i in range(0, loop_length, 2):
			child = parents[i].breed(parents[i+1], ratio_mean = ratio_mean)
			child.mutate(num_mutations = num_mutations)
			children.append(child)

		# Make sure these requirements are fulfilled
		assert self.dimensions == child.get_dimensions()
		assert self.num_individuals == len(children)
		
		# Set the current population to the children
		self.population = children
		return


	# Used for testing the sort_population() method
	#def set_fitness(self):
	#	for i, individual in enumerate(self.population):
	#		individual.set_fitness(i)


#g = Generation(num_individuals = 3, dimensions = [4,2,1], activation_hidden = 'relu')
#print(g)

#g.print_population()
#g.set_fitness()
#g.print_population()
#g.sort_population()
#g.print_population()