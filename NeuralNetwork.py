import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from numpy.random import uniform, choice

class NeuralNetwork:

	# Note: dimensions = [a, b, ..., z] means that the network has a nodes in the first hidden layer, b in the second, ... and z in the output layer
	def __init__(self, dimensions, num_inputs, activation_hidden, activation_output = 'sigmoid', randomize = True, weights = [], fitness = 0):
		self.dimensions = dimensions
		self.activation_hidden = activation_hidden
		self.activation_output = activation_output
		self.model = Sequential()
		self.fitness = fitness
		self.num_inputs = num_inputs
		
		self.model.add(Dense(units = dimensions[0], activation = activation_hidden, input_dim = num_inputs)) # First layer
		for nodes in dimensions[1:-1]:
			self.model.add(Dense(units = nodes, activation = activation_hidden))

		if len(dimensions) > 1:
			self.model.add(Dense(units = dimensions[-1], activation = activation_output)) # Last layer

		# Initialize the biases non-zero
		if randomize:
			weights = self.get_weights()
			new_weights = [uniform(low = -0.5, high = 0.5 , size = layer.shape[0]) if not np.any(layer) else layer for layer in weights]
			self.model.set_weights(new_weights)
			#print('weights =', weights)
			#print('new_weights = ', new_weights)
		
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

	def predict_binary(self, x):
		pred = float(self.model.predict(x = x))
		if pred < 0.5:
			return 0
		else:
			return 1

	def predict_multiclass(self, x):
		pred = self.model.predict(x = x) # Forward pass (resulting in vector with same dimension as number of output nodes)
		action = np.argmax(pred)
		return action

	def get_weights(self):
		return self.model.get_weights()

	# Crossover using arithmetic mean of ratio_mean of all weights and biases, (1-ratio_mean)/2 of the weights are kept from self, and another (1-ratio_mean)/2 is kept from NeuralNetwork2 
	def breed(self, NeuralNetwork2, ratio_mean = 0.5):

		w1_all = self.model.get_weights() 										# Weights of this NeuralNetwork
		w1_flattened = self.flatten()											# Flattened weights of this NeuralNetwork
		w2_all = NeuralNetwork2.getModel().get_weights() 						# Weights of NeuralNetwork2
		w2_flattened = NeuralNetwork2.flatten()									# Flattened weights of NeuralNetwork2

		num_weights = w1_flattened.shape[0]										# Total number of weights in the NeuralNetworks
		num_weights_mean = int(ratio_mean * num_weights)						# Number of weights to be used for arithmetic mean
		num_weights_keep1 = int((1-ratio_mean)/2 * num_weights)					# Number of weights to keep from self
		num_weights_keep2 = num_weights - num_weights_mean - num_weights_keep1	# Number of weights to keep from NeuralNetwork2

		#print('num_weights_mean = ', num_weights_mean)
		#print('num_weights_keep1 = ', num_weights_keep1)
		#print('num_weights_keep2 = ', num_weights_keep2)

		breed_types = [0]*num_weights_mean + [1]*num_weights_keep1 + [2]*num_weights_keep2
		random.shuffle(breed_types)

		new_weights_flattened = []
		for i, breed_type in enumerate(breed_types):
			
			if breed_type == 0: # Using arithmetic mean
				new_weights_flattened.append((w1_flattened[i] + w2_flattened[i])/2)

			elif breed_type == 1: # Keeping weight from self
				new_weights_flattened.append(w1_flattened[i])

			else: # Keeping weights from NeuralNetwork2
				new_weights_flattened.append(w2_flattened[i])

		new_weights = self.inverse_flatten(new_weights_flattened)

		NeuralNetwork_new = NeuralNetwork(dimensions = self.dimensions, num_inputs = self.num_inputs, activation_hidden = self.activation_hidden, 
			activation_output = self.activation_output, randomize = False, weights = new_weights)

		return NeuralNetwork_new



		#print('breed_type = ', breed_type)
		#rand = list(range(10))
		#random.shuffle(rand)
		#print(rand)


		#print('WEIGHTS_1 : ', w1_all)
		#print('WEIGHTS_2 : ', w2_all)
		#weights_all_new = []
		#for w1, w2 in zip(w1_all, w2_all):	
		#	mean_weights = (w1 + w2) / 2
		#	weights_all_new.append(mean_weights)

		#NeuralNetwork_new = NeuralNetwork(dimensions = self.dimensions, num_inputs = self.num_inputs, activation_hidden = self.activation_hidden, 
		#	activation_output = self.activation_output, randomize = False, weights = weights_all_new)
		#print('NEW WEIGHTS = ', NeuralNetwork_new.getModel().get_weights())
		#return NeuralNetwork_new

	def flatten(self):
		flattened = np.array([])
		
		for layer in self.get_weights():
			flattened = np.append(flattened, layer.flatten())
		
		return flattened

	# Takes a numpy-array 'flattened' (vector) with weights and transforms them into a list 'weights' of the same form as that from a keras sequential model
	def inverse_flatten(self, flattened):

		weights = []

		# Adding first layer weights
		start_pos = 0
		end_pos = start_pos + self.num_inputs*self.dimensions[0]
		new_shape = (self.num_inputs, self.dimensions[0])

		weights.append(np.reshape(flattened[start_pos:end_pos], newshape = new_shape))

		# Adding first layer biases
		start_pos = end_pos
		end_pos = start_pos + self.dimensions[0]

		weights.append(flattened[start_pos:end_pos])

		# Adding weights and biases of the remaining layers
		for layer in range(len(self.dimensions[1:])):
			
			#print('layer = ', layer)

			start_pos = end_pos
			end_pos = start_pos + self.dimensions[layer] * self.dimensions[layer+1]
			new_shape = (self.dimensions[layer], self.dimensions[layer+1])

			weights.append(np.reshape(flattened[start_pos:end_pos], newshape = new_shape))

			start_pos = end_pos
			end_pos = start_pos + self.dimensions[layer+1]

			weights.append(flattened[start_pos:end_pos])			

		return weights

	# Mutates this neural network by updating num_mutations weights with new values draw from a uniform distribution
	def mutate(self, num_mutations = 1):
		
		flattened = self.flatten()
		num_weights = flattened.shape[0]

		try:
			rand_ints = choice(list(range(num_weights)), num_mutations, replace = False)
		except:
			print('Exception: Cannot take a larger sample than population in np.random.choice. Please choose a smaller num_mutations.')
			return

		rand_unif = uniform(low = min(flattened), high = max(flattened) , size = num_mutations) # Draws from uniform distribution which depends on the min and max value of the weights and biases, i.e. flattened
		flattened[rand_ints] = rand_unif 														# Mutates flattened by updating the randomly chosen weights/biases
		new_weights = self.inverse_flatten(flattened) 											# Transforms the weight vector back to the desired format
		self.model.set_weights(new_weights)														# Updates the weights of self
		return





#nn = NeuralNetwork(dimensions = [3, 1], num_inputs = 2, activation_hidden = 'relu')

#print('Weights before = ', nn.get_weights(), type(nn.get_weights()))
#flattened = nn.flatten()
#print('Flattened = ', flattened, flattened.shape)
#new_weights = nn.inverse_flatten(flattened)
#print('new_weights = ', new_weights)
#nn2 = NeuralNetwork(dimensions = [3, 1], num_inputs = 2, activation_hidden = 'relu', activation_output = 'sigmoid', randomize = False, weights = new_weights)
#print('nn2 weights = ', nn2.get_weights())
#print('New = ', new)

#nn.mutate()
#print('Weights after = ', nn.get_weights(), type(nn.get_weights()))




#from keras import backend as K

#def my_init(shape, dtype=None):
#    return K.random_normal(shape, dtype=dtype)

#model.add(Dense(64, kernel_initializer=my_init))


#model1 = NeuralNetwork(dimensions = [5, 3, 1], activation_hidden = 'relu')
#model2 = NeuralNetwork(dimensions = [5, 3, 1], activation_hidden = 'relu')
#model1.breed(model2)
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