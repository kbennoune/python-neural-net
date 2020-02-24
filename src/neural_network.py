from __future__ import print_function
from __future__ import division
import numpy as np
import bigfloat
from gradient_descent_algorithm import GradientDescentAlgorithm
from normalized_batch_gradient_descent import NormalizedBatchGradientDescent

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class DescentCalculator:
    def __init__(self, feed_forward, calculate_deltas, gradient_descent):
        self.feed_forward = feed_forward
        self.calculate_deltas = calculate_deltas
        self.gradient_descent = gradient_descent

    def run(self, existing_weights, existing_biases, batch_data, batch_labels, batch_size):
        activations, outputs = self.feed_forward(existing_weights, existing_biases, batch_data, batch_labels)

        deltas = self.calculate_deltas(activations, outputs, existing_weights, batch_labels)

        weights, biases = self.gradient_descent(batch_size, deltas, existing_weights, activations, existing_biases)

        return weights, biases, activations, outputs

class NeuralNetwork:
    def __init__(self, topology, activation=sigmoid, use_bias_nodes=False, eta=3.0, algorithm_class=GradientDescentAlgorithm):
        self.use_bias_nodes = use_bias_nodes
        self.topology = topology
        self.effective_topology = self.__calc_effective_topology()

        self.activation = activation
        self.weights, self.biases, self.activations = self.__initial_attributes()
        self.algorithm = algorithm_class(topology=topology, eta=eta)
        self.descent_calculator = DescentCalculator(self.algorithm.feed_forward, self.algorithm.calculate_deltas, self.algorithm.gradient_descent)

    def __initial_attributes(self):
        weights = [np.random.randn(j, i).T for i, j in zip(self.effective_topology[:-1], self.effective_topology[1:])]
        biases = [np.random.randn(i, 1).T for i in self.effective_topology[1:]]
        activations = map(lambda shape: np.zeros(shape), self.topology[1:])

        return weights, biases, activations

    def __calc_effective_topology(self):
        if self.use_bias_nodes:
            effective_topology = [self.topology[0]] + [n+1 for n in self.topology[1:-1]] + [self.topology[-1]]
        else:
            effective_topology = self.topology

        return effective_topology

    def train(self, data, indexed_labels, test_data=None, test_labels=None, epochs=200, batch_size=1000):
        labels = self.vectorize(indexed_labels)

        for epoch in range(epochs):
            self.run_training(data,labels,batch_size=batch_size)

            #evaluate
            final_activations, _ = self.descent_calculator.feed_forward(self.weights, self.biases, data, labels)

            results = np.argmax(final_activations[-1], axis=1)
            cost = self.cost(results, indexed_labels.flatten())
            from collections import Counter
            # print("The weights of the predictions are: {}".format(Counter(results)))
            print("Training Error: {}% for epoch {}".format(cost * 100, epoch))

            if test_data is not None:
                test_activations, _ = self.descent_calculator.feed_forward(self.weights, self.biases, test_data, self.vectorize(test_labels))
                test_results = np.argmax(test_activations[-1], axis=1)
                test_cost = self.cost(test_results, test_labels.flatten())
                # print("The weights of the test predictions are: {}".format(Counter(test_results)))
                print("Test Error: {}% for epoch {}\n".format(test_cost * 100, epoch))

            self.algorithm.finalize_epoch()
        return True

    def run_training(self, data, labels, batch_size=200):
        num_of_examples = data.shape[0]
        permutation = np.random.permutation(num_of_examples)

        data = data[permutation]
        labels = labels[permutation]

        batches_data = [data[k:k + batch_size, :] for k in range(0, num_of_examples, batch_size)]
        batches_labels = [labels[k:k + batch_size, :] for k in range(0, num_of_examples, batch_size)]

        for batch_data, batch_labels in zip(batches_data, batches_labels):

            self.weights, self.biases, self.activations, self.outputs = \
              self.descent_calculator.run(self.weights, self.biases, batch_data, batch_labels, batch_size)
        return True

    def cost(self, predicted, target):
        return float(np.sum(predicted != target)) / len(predicted)

    def vectorize(self,labeled_data):
        options = np.unique(labeled_data)

        vectorized_data = np.zeros([labeled_data.shape[0],options.shape[0]])
        for idx, val in enumerate(labeled_data):
            pos = np.nonzero(options == val)[0][0]
            vectorized_data[idx, pos] = 1

        return vectorized_data
