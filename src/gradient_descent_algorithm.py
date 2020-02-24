import numpy as np
from functions import sigmoid, sigmoid_derivative

class GradientDescentAlgorithm(object):
    def __init__(self,eta=3.0,topology=[]):
        self.eta = eta
        self.topology = topology
        pass

    def finalize_epoch(self):
        pass

    def gradient_descent(self, num_of_examples, deltas, weights, activations, biases):
        new_weights = np.zeros(len(weights)).tolist()
        new_biases = np.zeros(len(biases)).tolist()
        scalar = float(self.eta)/num_of_examples

        for layer, output in reversed(list(enumerate(deltas))):
            new_weights[layer] = weights[layer] - scalar * np.dot(deltas[layer].T, activations[layer]).T
            new_biases[layer] = biases[layer] - scalar * np.sum(deltas[layer], axis=0)

        return new_weights, new_biases

    # calculate initial error and back_propagate
    def calculate_deltas(self, activations, outputs, weights, labels):
        deltas = np.zeros(len(outputs)).tolist()
        for layer, output in reversed(list(enumerate(outputs))):
            if layer == (len(outputs) - 1):
                first_term = activations[-1] - labels
            else:
                first_term = np.dot(deltas[layer+1], weights[layer+1].T)

            second_term = sigmoid_derivative(output)

            deltas[layer] = np.multiply(first_term, second_term)

        return deltas

    def feed_forward(self, weights, biases, data, labels):
        # Add data as first layer for activations
        new_activations = [data]
        new_outputs = []
        number_of_labels = labels.shape[0]

        for idx, weight in enumerate(weights):
            outputs, activations = self.feed_forward_layer(idx,weight,new_activations[idx],biases[idx])
            new_outputs.append(outputs)
            new_activations.append( activations )

        return new_activations, new_outputs

    def feed_forward_layer(self,layer,weight,inputs,bias):
        outputs = self.calculate_outputs(layer,inputs,weight,bias)
        activations = self.calculate_activations(layer,outputs)

        return outputs, activations

    def calculate_outputs(self,layer, inputs,weight,bias):
        outputs = np.dot(inputs, weight) + bias

        return outputs


    def calculate_activations(self,layer,outputs):
        return sigmoid(outputs)
