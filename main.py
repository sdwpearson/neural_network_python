"""
Main entry point for the neural network code
"""

#!/usr/bin/python
import network_components

def print_network(network):
	for i in range(len(network.layers)):
		print "layer " + `i`
		for j in range(len(network.layers[i].neurons)):	
			print network.layers[i].neurons[j].weights	
			print network.layers[i].neurons[j].bias	
			print network.layers[i].neurons[j].output

num_layers = 3
neurons_per_layer = [2,3,2]
input_dimension = 2
output_dimension = 2

network = network_components.Network(num_layers, neurons_per_layer, input_dimension)

print_network(network)

inputs = [1.3,0.09]
network.update(inputs)

print 

print_network(network)

print network.backpropagate([2, 1], inputs)