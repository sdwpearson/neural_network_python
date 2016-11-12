"""
Code containing the neurons, layers, and network classes
"""
import math
import random
import numpy

#------------------------------------------------
# sigmoid(x) - Returns x when sigmoided
#------------------------------------------------
def sigmoid(x):
	return 1 / (1 + math.exp(-x))

#------------------------------------------------
# sigmoid_primed(x) - Returns the derivative of 
# the sigmoid funciton when x in input
#------------------------------------------------
def sigmoid_primed(x):	
	return sigmoid(x) * (1-sigmoid(x))


#================================================
# Neuron - The neuron class which models an 
# individual neuron used to form layers
#
# self.weights - A vector of the weights for the 
# 				 neuron. It is effectively the 
#      			 weights of this layer * activations
# 				 of the previous layer
# self.bias - The bias of the neuron
# self.output - Effectively 'a'. It is sigmoid(z)
#================================================
class Neuron:
	# Inititailizes the neuron with a given bias and set of weights
	def __init__(self, weights, bias):
		self.inputs = 0.0
		self.weights = 0.0
		self.weights = weights
		self.bias = 0.0
		self.bias = bias
		self.output = 0.0

	# Updates the neuron's output (a) given an input to the neuron
	def update(self, inputs_to_neuron):
		self.inputs = inputs_to_neuron
		weight_sum = 0
		for i in range(len(self.weights)):
			weight_sum += self.weights[i] * self.inputs[i]
		self.output = sigmoid(weight_sum + self.bias)

	# Calculates the error assuming this neuron is on the last layer
	def initial_error(self, expected_output):
		dCda = self.output - expected_output
		return dCda * sigmoid_primed(self.get_z())

	def get_z(self):
		weight_sum = 0
		for i in range(len(self.weights)):
			weight_sum += self.weights[i] * self.inputs[i]
		return weight_sum + self.bias

	def get_a(self):
		weight_sum = 0
		for i in range(len(self.weights)):
			weight_sum += self.weights[i] * self.inputs[i]
		return sigmoid(weight_sum + self.bias)


#================================================
# Layer - made up of individual neurons and used
# to create the Network class.
#
# self.neurons - a list of the class Neuron
#================================================
class Layer:
	# Initialize each neuron with a random set of values
	def __init__(self, num_neurons, prev_num_neurons):
		self.neurons = []
		for i in range(num_neurons):
			weights = []
			for i in range(prev_num_neurons):
				weights.append(random.random())
			self.neurons.append(Neuron(weights, random.random()))

	# Goes through each neuron and updates their outputs
	def update(self, inputs):
		for i in range(len(self.neurons)):
			self.neurons[i].update(inputs)

	# Calculates the initial error of the layer assuming it is the final layer
	def initial_error(self, expected_output):
		errors = []
		for i in range(len(self.neurons)):
			errors.append(self.neurons[i].initial_error(expected_output[i]))
		return errors

	# Retrives all the z's (the output before it is sigmoided) in a vector format
	def get_z(self):
		z = []
		for i in range(len(self.neurons)):
			z.append(self.neurons[i].get_z())
		return numpy.asarray(z)

	# Retrives all the a's (the output) in a vector format
	def get_a(self):
		a = []
		for i in range(len(self.neurons)):
		#	a.append(sigmoid(sum(self.neurons[i].weights) + self.neurons[i].bias))
			a.append(self.neurons[i].output)	
		return numpy.asarray(a)

	# Retrieves the weight matrix from the neurons in the layer
	# Rows are each neuron in current layer
	# Columns are the weights applied to the current neuron from previous layer
	def get_weight_matrix(self):
		weight_matrix = []
		for i in range(len(self.neurons)):
			weight_matrix.append(self.neurons[i].weights)
		return numpy.asarray(weight_matrix).astype(float)

	# Retrives all the biases in a vector format
	def get_biases(self):
		biases = []
		for i in range(len(self.neurons)):
			biases.append(self.neurons[i].bias)
		return numpy.asarray(biases)

	def set_weights(self, weight_matrix):
		for i in range(len(self.neurons)):
			self.neurons[i].weights = weight_matrix[i]


	def set_biases(self, bias_vector):
		for i in range(len(self.neurons)):
			self.neurons[i].bias = bias_vector[i]



#================================================
# Network - Made up of individual layers
#
# self.layers - A list of the class Layer
# self.dCdb - A list of matrices that contain the
# 			  derivative of the cost function with
#  			  respect to the individual biases
# self.dCdw - A list of weight matrices that 
# 		 	  contain the derivatives of the cost
# 			  function with respect to the weights
#================================================
class Network:
	# Initialize the network with random values in the neurons in each of the layers.
	def __init__(self, num_layers, neurons_per_layer, input_dimension):
		self.layers = []
		for i in range(num_layers):
			if (i == 0):
				prev_neuron = input_dimension
			else:
				prev_neuron = neurons_per_layer[i-1]
			self.layers.append(Layer(neurons_per_layer[i], prev_neuron))

	# Update the entire network when given a set of inputs. The neurons in 
	# the first layer each use their own input and the inputs aren't shared
	# across the neurons of the first layer. That is, input 1 is only used by
	# neuron 1. It is assumed the dimension of inputs is the same as that of the 
	# first layer.
	def update(self, inputs):
		prev_outputs = []
		for i in range(len(self.layers)):
			if (i == 0):
				self.layers[i].update(inputs)
			else:
				#for j in range(len(self.layers[i-1].neurons)):
				#	prev_outputs.append(self.layers[i-1].neurons[j].output)
				prev_outputs = self.layers[i-1].get_a()
				self.layers[i].update(prev_outputs)

	# Calculates the error of the last layer
	def initial_error(self, expected_output):
		return self.layers[-1].initial_error(expected_output)

	# Performs the entire backpropagation function, storing the two derivatives in dCdb and dCdw
	def backpropagate(self, expected_output, inputs):
		error = []
		error.append(numpy.transpose(numpy.asarray(self.initial_error(expected_output))))
		vec_sigmoid = numpy.vectorize(sigmoid)
		vec_sigmoid_primed = numpy.vectorize(sigmoid_primed)
		# Go through each layer and find all of the errors for the individual neurons
		for i in range(len(self.layers)):
			if(i != 0):
				if(i == 1):
					first_part = numpy.dot(numpy.transpose(self.layers[-1].get_weight_matrix()),error[0])
					second_part = numpy.asarray(vec_sigmoid_primed(self.layers[-2].get_z()))
					error.append(numpy.multiply(first_part,second_part))
				else:
					curr_layer = (-1)*(i + 1)
					prev_layer = (-1) * i
					first_part = numpy.dot(numpy.transpose(self.layers[prev_layer].get_weight_matrix()),error[i-1])
					second_part = vec_sigmoid_primed(self.layers[curr_layer].get_z())
					error.append(numpy.multiply(first_part, second_part))
		error.reverse()
		self.dCdb = error
		weight_matrices = []
		# Go through all the layers and find dCdw for each weight
		for i in range(len(self.layers)):
			weight_matrices.append(self.layers[i].get_weight_matrix())
			if(i == 0):
				dimensions = weight_matrices[0].shape
				for j in range(dimensions[0]):
					for k in range(dimensions[1]):
						weight_matrices[0][j][k] = inputs[k]*error[0][j]
			else:
				dimensions = weight_matrices[i].shape
				for j in range(dimensions[0]):
					for k in range(dimensions[1]):
						weight_matrices[i][j][k] = self.layers[i-1].neurons[k].output*error[i][j]
		self.dCdw = weight_matrices
		# Return the error list of matrices because why not
		return error

	# Perform gradient descent given a set of errors calculated from backpropagation
	def gradient_descent(self, errors, learning_rate, num_batches):
		for i in range(len(self.layers) - 1):
			bias_sum = 0.0	
			weight_sum = 0.0
			weight_matrix = self.layers[len(self.layers)-(i+1)].get_weight_matrix()
			bias_vector = self.layers[len(self.layers)-(i+1)].get_biases()

			for j in range(len(errors[len(self.layers)-(i+1)])): # Go from last layers in
				bias_sum += errors[len(self.layers)-(i+1)][j]
				a = self.layers[len(self.layers)-(i+1)].get_a()
				weight_sum += errors[len(self.layers)-(i+1)][j]*a[j]

			weight_sum = weight_sum*(float(learning_rate)/float(num_batches))*(-1)
			bias_sum = bias_sum*(learning_rate/num_batches)*(-1)
			weight_matrix += weight_sum
			bias_vector += bias_sum
			self.layers[len(self.layers)-(i+1)].set_weights(weight_matrix)
			self.layers[len(self.layers)-(i+1)].set_biases(bias_vector)

	# Get the output of the network
	def get_output(self):
		return self.layers[-1].get_a()













