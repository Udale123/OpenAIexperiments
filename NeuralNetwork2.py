# this is going to be an implementation of a NN using matrics instead of loops
import numpy as np
import math


class NeuralNetwork:
    def __init__(self, layers=[2, 3, 1], activation='sigmoid'):
        self.layers = layers
        self.activation = activation

        self.NeuronOutputs = [np.zeros(i) for i in layers]
        self.NeuronBias = [np.random.randn(i) for i in layers]
        self.weights = [np.random.randn(layers[i + 1], layers[i]) for i in range(len(layers) - 1)]

    def activate(self, value):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-value))
        elif self.activation == 'tanh':
            return np.tanh(value)

    def feed_forward(self, inputs):
        self.NeuronOutputs[0] = self.activate(np.array(inputs))
        for i in range(len(self.NeuronOutputs[1:])):
            self.NeuronOutputs[i + 1] = self.activate(
                np.matmul(self.weights[i], self.NeuronOutputs[i]) + self.NeuronBias[i + 1])
        return self.NeuronOutputs[-1]

    def mutate(self, rate=0.05):
        for i in range(len(self.NeuronBias)):
            r = np.random.choice([True, False], p=[1 - rate, rate], size=self.NeuronBias[i].shape)
            self.NeuronBias[i] = np.where(r, self.NeuronBias[i], np.random.randn())
        for j in range(len(self.weights)):
            r = np.random.choice([True, False], p=[1 - rate, rate], size=self.weights[j].shape)
            self.weights[j] = np.where(r, self.weights[j], np.random.randn())

    def duplicate(self):
        child = NeuralNetwork(self.layers)
        child.weights = self.weights
        child.NeuronBias = self.NeuronBias
        return child

    def cross_over(self, spouse, rate=0.05):
        child = NeuralNetwork(self.layers)
        for i in range(len(self.NeuronBias)):
            r = np.random.choice([True, False], p=[1 - rate, rate], size=self.NeuronBias[i].shape)
            child.NeuronBias[i] = np.where(r, self.NeuronBias[i], spouse.NeuronBias[i])
        for j in range(len(self.weights)):
            r = np.random.choice([True, False], p=[1 - rate, rate], size=self.weights[j].shape)
            child.weights[j] = np.where(r, self.weights[j], spouse.weights[j])
        return child

