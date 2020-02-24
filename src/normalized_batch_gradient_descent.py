from gradient_descent_algorithm import GradientDescentAlgorithm
import numpy as np
from functions import sigmoid, sigmoid_derivative


class NormalizedBatchGradientDescent(GradientDescentAlgorithm):
    def __init__(self,eta=3.0,topology=[]):
        self.beta = [np.zeros(len) for len in topology[1:]]
        self.gamma = [np.ones(len) for len in topology[1:]]

        self.means = [[] for len in topology[1:]]
        self.deviations = [[] for len in topology[1:]]
        super(NormalizedBatchGradientDescent, self).__init__(eta=eta, topology=topology)

    def calculate_outputs(self,layer,inputs,weight,bias):
        outputs = super(NormalizedBatchGradientDescent,self).calculate_outputs(layer,inputs,weight,bias)
        mean, deviation = self._calculate_batch_stats(outputs)
        self.means[layer].append(mean)
        self.deviations[layer].append(deviation)

        outputs_hat = self.normalize(outputs,mean,deviation,layer)
        return outputs_hat

    def normalize(self,outputs,mean,deviation,layer):
        return self.gamma[layer] * self.transform(outputs,mean,deviation) + self.beta[layer]

    def transform(self, outputs,mean,deviation):
        return (outputs - mean)/deviation

    def _calculate_batch_stats(self,outputs):
        epsilon = 1e-10
        num_outputs = outputs.shape[0]
        mean = np.mean(outputs,axis=0)
        # import pdb; pdb.set_trace()
        deviation = np.sqrt(epsilon + sum((outputs - mean) ** 2)/num_outputs)

        return mean, deviation

    def calculate_deltas(self, activations, outputs, weights, labels):
        # deltas = super(NormalizedBatchGradientDescent, self).calculate_deltas(activations, outputs, weights, labels)

        deltas = np.zeros(len(outputs)).tolist()
        dgammas = np.zeros(len(self.gamma)).tolist()
        dbetas = np.zeros(len(self.beta)).tolist()

        for layer, output in reversed(list(enumerate(outputs))):
            if layer == (len(outputs) - 1):
                first_term = activations[-1] - labels
            else:
                dout = deltas[layer+1]
                x = outputs[layer+1]
                epsilon = 1e-15
                num_outputs = dout.shape[0]

                mean = np.mean(self.means[layer+1],axis=0)
                deviation = (1./num_outputs)*np.sum((self.means[layer+1]-mean)**2, axis = 0)#np.mean(self.deviations[layer+1],axis=0)
                gamma = self.gamma[layer+1]
                t = (deviation + epsilon) ** (-1./2.)

                dout_hat1 = num_outputs * dout
                dout_hat2 = -np.sum(dout, axis=0)
                dout_hat3 = - (t ** 2) * (x - mean) * np.sum(dout*(x - mean), axis=0)
                dout_hat = ((gamma * t)/num_outputs) * (dout_hat1 + dout_hat2 + dout_hat3)

                first_term = np.dot(dout_hat, weights[layer+1].T)

            second_term = sigmoid_derivative(output)

            deltas[layer] = np.multiply(first_term, second_term)
            dgammas[layer] = np.sum(deltas[layer] * self.transform(outputs[layer], self.means[layer][-1],  self.deviations[layer][-1]), axis=0)
            dbetas[layer] = np.sum(deltas[layer], axis=0)

        new_gamma = [self.gamma[layer] - dgammas[layer] for layer in range(len(self.gamma)) ]
        new_beta = [self.beta[layer] - dbetas[layer] for layer in range(len(self.beta)) ]

        # print([np.linalg.norm(b-nb) for b,nb in zip(self.beta,new_beta)])
        # norms = [np.linalg.norm(g-ng) for g,ng in zip(self.gamma,new_gamma)]
        #
        # print(norms)
        # if norms[2] > 100:
        #     import pdb; pdb.set_trace()

        self.beta = new_beta
        self.gamma = new_gamma

        return deltas

    def finalize_epoch(self):
        self.means = [[] for len in self.topology[1:]]
        self.deviations = [[] for len in self.topology[1:]]
