import numpy as np

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
