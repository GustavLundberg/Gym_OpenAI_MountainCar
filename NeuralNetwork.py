import numpy as np
from keras.models import Sequential
from keras.layers import Dense

class NeuralNetwork:

	# Note: dimensions = [a, b, ..., z] means that the network has a nodes in the first hidden layer, b in the second, ... and z in the output layer
	def __init__(self, dimensions, activation_hidden, activation_output = 'sigmoid', randomize = True, weights = [], fitness = 0):
		self.dimensions = dimensions
		self.activation_hidden = activation_hidden
		self.activation_output = activation_output
		self.model = Sequential()
		self.fitness = fitness
		
		
		self.model.add(Dense(units = dimensions[0], activation = activation_hidden, input_dim = 4)) # First layer
		for nodes in dimensions[1:-1]:
			self.model.add(Dense(units = nodes, activation = activation_hidden))

		self.model.add(Dense(units = dimensions[-1], activation = activation_output)) # Last layer
		
		if not randomize:
			self.model.set_weights(weights) # Set the weights

	def __repr__(self):
		return 'NeuralNetwork object with dimensions ' + str(self.dimensions) + ' and activation_hidden ' + str(self.activation_hidden)

	def getModel(self):
		return self.model

	def set_fitness(self, fitness):
		self.fitness = fitness
		return

	def get_fitness(self):
		return self.fitness

	def get_dimensions(self):
		return self.dimensions

	def custom_initializer(self, shape, weights, dtype = None):
		return weights

	def predict(self, x):
		pred = float(self.model.predict(x = x))
		if pred < 0.5:
			return 0
		else:
			return 1

	def get_weights(self):
		return self.model.get_weights()

	# Crossover using arithmetic mean of all weights and biases
	def breed(self, NeuralNetwork2):

		w1_all = self.model.get_weights() 					# Weights of this NeuralNetwork
		w2_all = NeuralNetwork2.getModel().get_weights() 	# Weights of NeuralNetwork2
		#print('WEIGHTS_1 : ', w1_all)
		#print('WEIGHTS_2 : ', w2_all)
		weights_all_new = []
		for w1, w2 in zip(w1_all, w2_all):	
			mean_weights = (w1 + w2) / 2
			weights_all_new.append(mean_weights)

		NeuralNetwork_new = NeuralNetwork(dimensions = self.dimensions, activation_hidden = self.activation_hidden, 
			activation_output = self.activation_output, randomize = False, weights = weights_all_new)
		#print('NEW WEIGHTS = ', NeuralNetwork_new.getModel().get_weights())
		return NeuralNetwork_new


	# Det kanske ær bættre att bara sætta upp en regel: sampla ett integer och sen låta det motsvara en plats i den totala vikt-listan. Genom att anvænda lite modulo och sånt! Typ som nær man ska væxla pengar. Det kommer ju såklart att bero på dimensions, dvs hur många noder och dærmed vikter det finns i lagrena.
	def flatten(self):
		flattened = []
		for layer in self.get_weights():
			flat = layer.flatten()
			print('flat = ', flat, flat.shape)
			flattened.append(layer.flatten())

		return flattened

	#def mutate(self):





#from keras import backend as K

#def my_init(shape, dtype=None):
#    return K.random_normal(shape, dtype=dtype)

#model.add(Dense(64, kernel_initializer=my_init))


model1 = NeuralNetwork(dimensions = [5, 3, 1], activation_hidden = 'relu')
model2 = NeuralNetwork(dimensions = [5, 3, 1], activation_hidden = 'relu')
model1.breed(model2)
#print(model)
#example = np.array([1, 2, 3, 4])
#example = np.reshape(example, [1, 4])
#print(example, example.shape)
#p = model.getModel().predict(x = example)
#p = float(p)
#print('Prediction: ', p)
#w = model.getModel().get_weights()
#print('Weights: ', w)

# a = 2 | 10 # Bitwise operation
# print('a = ', a)