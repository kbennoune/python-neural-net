#!/usr/local/bin/python
import sys
import numpy
import os
import gzip
from struct import unpack

sys.path.append('src')

from neural_network import NeuralNetwork
from normalizer import Normalizer

DATA_FOLDER = 'data'

TRAIN = { "data" : 'train-images-idx3-ubyte.gz', "labels" : 'train-labels-idx1-ubyte.gz'}
TEST = { "data" :'t10k-images-idx3-ubyte.gz', "labels" : 't10k-labels-idx1-ubyte.gz'}


def main():
    numpy.random.seed(3)

    if not os.path.exists("cache"):
        os.makedirs("cache")

    if os.path.exists("cache/training_data.npy"):
        training_data = numpy.load('cache/training_data.npy')
    else:
        training_data = read_images(TRAIN['data'], DATA_FOLDER)
        numpy.save('cache/training_data',training_data)

    if os.path.exists("cache/training_labels.npy"):
        training_labels = numpy.load('cache/training_labels.npy')
    else:
        training_labels = read_labels(TRAIN['labels'], DATA_FOLDER)
        numpy.save('cache/training_labels',training_labels)

    training_data = training_data.reshape(training_data.shape[0],-1)

    number_of_inputs = training_data.shape[1]
    number_of_labels = numpy.unique(training_labels).size
    topology = [number_of_inputs, 50, 30, number_of_labels]

    if os.path.exists("cache/test_data.npy"):
        test_data = numpy.load('cache/test_data.npy')
    else:
        test_data = read_images(TEST['data'], DATA_FOLDER)
        numpy.save('cache/test_data',test_data)

    if os.path.exists("cache/test_labels.npy"):
        test_labels = numpy.load('cache/test_labels.npy')
    else:
        test_labels = read_labels(TEST['labels'], DATA_FOLDER)
        numpy.save('cache/test_labels',test_labels)

    test_data = test_data.reshape(test_data.shape[0],-1)

    normalized = Normalizer(training_data=training_data,test_data=test_data)

    for i in range(1):
        print("Running {}th time...".format(i))
        neural_network = NeuralNetwork(topology)
        neural_network.train(normalized.training_data(),training_labels,test_data=normalized.test_data(), test_labels=test_labels)

def read_images(file,directory):
    file_location = os.path.join(directory, file)
    with gzip.open(file_location, 'rb') as images:
        x0 = images.read(4)
        number_of_images = images.read(4)
        number_of_images = unpack('>I', number_of_images)[0]
        rows = images.read(4)
        rows = unpack('>I', rows)[0]
        cols = images.read(4)
        cols = unpack('>I', cols)[0]

        x = numpy.zeros((number_of_images, rows, cols), dtype=numpy.uint8)

        for i in range(number_of_images):
            if i % int(number_of_images / 10) == int(number_of_images / 10) - 1:
                print("Reading images progress: {}%".format(int(100 * (i + 1) / number_of_images)))
            for row in range(rows):
                for col in range(cols):
                    tmp_pixel = images.read(1)  # Just a single byte
                    tmp_pixel = unpack('>B', tmp_pixel)[0]
                    x[i][row][col] = tmp_pixel

    return x

def read_labels(file_name, data_folder):
    file_location = os.path.join(data_folder, file_name)
    with gzip.open(file_location, 'rb') as labels:
        labels.read(4)
        number_of_labels = labels.read(4)
        number_of_labels = unpack('>I', number_of_labels)[0]
        y = numpy.zeros((number_of_labels, 1), dtype=numpy.uint8)

        for i in range(number_of_labels):
            tmp_label = labels.read(1)
            y[i] = unpack('>B', tmp_label)[0]

    return y

if __name__ == '__main__':

    main()
