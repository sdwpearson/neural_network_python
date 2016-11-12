"""
Main entry point for the neural network code
"""

#!/usr/bin/python
import network_components

def print_network(network):
	for i in range(len(network.layers)):
		print "layer " + `i`
		for j in range(len(network.layers[i].neurons)):	
			print "weights: " + str(network.layers[i].neurons[j].weights)	
			print "inputs: " + str(network.layers[i].neurons[j].inputs)	
			print "bias: " + str(network.layers[i].neurons[j].bias)	
			print "output: " + str(network.layers[i].neurons[j].output)

def complete_network_cycle(network, inputs, train, expected_outputs, learning_rate, num_batches):
	"""print "\ninitial network"
	print_network(network)
	weights = []
	weights.append([0.1937])
	network.layers[0].set_weights(weights)	
	network.layers[0].set_biases([0.1968])
	weights2 = []
	weights2.append([0.2965])
	network.layers[1].set_weights(weights2) 
	network.layers[1].set_biases([0.4914])
	print_network(network)"""
	network.update(inputs)
	##print "\nafter update"
	##print_network(network)
	# Are we training the network?
	if(train):
		errors = network.backpropagate(expected_outputs, inputs)
		##print "\nafter backpropagation"
		##print_network(network)
		##print "errors: " + str(errors)
		network.gradient_descent(errors, learning_rate, num_batches)
		##print "\nafter gradient_descent"
		##print_network(network)
	return network.get_output()		

TRAIN = 1
TEST = 0
num_layers = 3
neurons_per_layer = [2,3,1]
input_dimension = 2
output_dimension = 1
learning_rate = 10
num_batches = 4
num_epochs = 10

network = network_components.Network(num_layers, neurons_per_layer, input_dimension)

# Train XOR Function
# 
# x1 | x2 | output
#  0   0      0
#  0   1      1
#  1   1      0
#  1   0      1
print "TRAINING:\n"
for i in range(num_epochs):
	print "EPOCH " + str(i)
	#print complete_network_cycle(network, [1.0], TRAIN, [0.0], learning_rate, num_batches)
	print complete_network_cycle(network, [0.0,0.0], TRAIN, [0.0], learning_rate, num_batches)
	print complete_network_cycle(network, [0.0,1.0], TRAIN, [1.0], learning_rate, num_batches)
	print complete_network_cycle(network, [1.0,0.0], TRAIN, [1.0], learning_rate, num_batches)
	print complete_network_cycle(network, [1.0,1.0], TRAIN, [0.0], learning_rate, num_batches)

print "\nTESTING \n"

print "expected = 0 got:" + str(complete_network_cycle(network, [0.0,0.0], TEST, [0.0], learning_rate, num_batches))
print "expected = 1 got:" + str(complete_network_cycle(network, [0.0,1.0], TEST, [1.0], learning_rate, num_batches))
print "expected = 0 got:" + str(complete_network_cycle(network, [1.0,1.0], TEST, [0.0], learning_rate, num_batches))
print "expected = 1 got:" + str(complete_network_cycle(network, [1.0,0.0], TEST, [1.0], learning_rate, num_batches))








