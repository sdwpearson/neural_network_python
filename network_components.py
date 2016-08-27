"""
Code containing the neurons, layers, and network classes
"""
import math
import random
import numpy

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def sigmoid_primed(x):	
	return sigmoid(x) * (1-sigmoid(x))

class Neuron:
	def __init__(self, weights, bias):
		self.weights = weights
		self.bias = bias
		self.output = 0

	def update(self, inputs):
		for i in range(len(self.weights)):
			self.weights[i] = self.weights[i] * inputs[i]
		self.output = sigmoid(sum(self.weights) + self.bias)

	def initial_error(self, expected_output):
		dCda = self.output - expected_output
		return dCda * sigmoid_primed(sum(self.weights) + self.bias)


class Layer:
	def __init__(self, num_neurons, prev_num_neurons):
		# initialize each neuron with a random set of values
		self.neurons = []
		for i in range(num_neurons):
			self.neurons.append(Neuron(random.sample(xrange(10), prev_num_neurons), random.random()))

	def update(self, inputs):
		for i in range(len(self.neurons)):
			self.neurons[i].update(inputs)

	def initial_error(self, expected_output):
		errors = []
		for i in range(len(self.neurons)):
			errors.append(self.neurons[i].initial_error(expected_output[i]))
		return errors

	def get_weight_matrix(self):
		weight_matrix = []
		for i in range(len(self.neurons)):
			weight_matrix.append(self.neurons[i].weights)
		return numpy.asarray(weight_matrix)

	def get_z(self):
		z = []
		for i in range(len(self.neurons)):
			z.append(sum(self.neurons[i].weights) + self.neurons[i].bias)
		return numpy.asarray(z)


class Network:
	def __init__(self, num_layers, neurons_per_layer, input_dimension):
		# initialize a given number of layers
		self.layers = []
		for i in range(num_layers):
			if (i == 0):
				prev_neuron = input_dimension
			else:
				prev_neuron = neurons_per_layer[i-1]
			self.layers.append(Layer(neurons_per_layer[i], prev_neuron))

	def update(self, inputs):
		prev_outputs = []
		for i in range(len(self.layers)):
			if (i == 0):
				self.layers[i].update(inputs)
			else:
				for j in range(len(self.layers[i-1].neurons)):
					prev_outputs.append(self.layers[i-1].neurons[j].output)
				self.layers[i].update(prev_outputs)

	def initial_error(self, expected_output):
		return self.layers[-1].initial_error(expected_output)

	def backpropagate(self, expected_output, inputs):
		error = []
		error.append(numpy.transpose(numpy.asarray(self.initial_error(expected_output))))
		vec_sigmoid = numpy.vectorize(sigmoid)
		vec_sigmoid_primed = numpy.vectorize(sigmoid_primed)
		for i in range(len(self.layers)):
			if(i != 0):
				if(i == 1):
					first_part = numpy.dot(numpy.transpose(self.layers[-1].get_weight_matrix()),error[0])
					second_part = numpy.asarray(vec_sigmoid_primed(self.layers[-2].get_z()))
					print first_part
					print second_part
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
		return error











