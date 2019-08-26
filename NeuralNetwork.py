import numpy as np
from keras.models import Sequential
from keras.layers import Dense

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

		NeuralNetwork_new = NeuralNetwork(dimensions = self.dimensions, num_inputs = self.num_inputs, activation_hidden = self.activation_hidden, 
			activation_output = self.activation_output, randomize = False, weights = weights_all_new)
		#print('NEW WEIGHTS = ', NeuralNetwork_new.getModel().get_weights())
		return NeuralNetwork_new


	# Det kanske ær bættre att bara sætta upp en regel: sampla ett integer och sen låta det motsvara en plats i den totala vikt-listan. Genom att anvænda lite modulo och sånt! Typ som nær man ska væxla pengar. Det kommer ju såklart att bero på dimensions, dvs hur många noder och dærmed vikter det finns i lagrena.
	def flatten(self):
		
		flattened = np.array([])
		
		for layer in self.get_weights():
			print('Shape of layer = ', layer.shape)
			flattened = np.append(flattened, layer.flatten())


		return flattened

	def inverse_flatten(self, flattened):

		weights = []
		#for i in range(2 * len(self.dimensions)):
		
		# First layer
		start_pos = 0
		end_pos = start_pos + self.num_inputs*self.dimensions[0]
		new_shape = (self.num_inputs, self.dimensions[0])

		weights.append(np.reshape(flattened[start_pos:end_pos], newshape = new_shape))

		start_pos = end_pos
		end_pos = start_pos + self.dimensions[0]

		weights.append(flattened[start_pos: (self.num_inputs*self.dimensions[0] + self.dimensions[0])])


		# Second layer
		start_pos = end_pos
		end_pos = start_pos + self.dimensions[0] * self.dimensions[1]
		new_shape = (self.dimensions[0], self.dimensions[1])

		weights.append(np.reshape(flattened[start_pos:end_pos], newshape = new_shape))

		start_pos = end_pos
		end_pos += self.dimensions[0]

		weights.append(flattened[start_pos: (self.num_inputs*self.dimensions[0] + self.dimensions[0])])		





		# Second layer
		start_pos += 
		weights.append(np.reshape(flattened[(self.num_inputs*self.dimensions[0] + self.dimensions[0]) : (self.num_inputs*self.dimensions[0] + self.dimensions[0])], 
			newshape = (self.num_inputs, self.dimensions[0])))

		print('weights = ', weights)

		#flattened[(self.num_inputs*self.dimensions[0]) : (self.num_inputs*self.dimensions[0] + self.dimensions[0])
		#np.reshape(flattened[(self.num_inputs*self.dimensions[0]) : (self.num_inputs*self.dimensions[0] + self.dimensions[0])], 
		#	newshape = (self.dimensions[0]))
		
		#print('weight_matrix = ', weight_matrix)
		#return weight_matrix

	#def mutate(self):


nn = NeuralNetwork(dimensions = [3, 1], num_inputs = 2, activation_hidden = 'relu')

print('Weights before = ', nn.get_weights(), type(nn.get_weights()))
flattened = nn.flatten()
print('Flattened = ', flattened, flattened.shape)
new = nn.inverse_flatten(flattened)
#print('New = ', new)






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