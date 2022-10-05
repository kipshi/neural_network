from util import *
import numpy as np

class Layer:
    def __init__(self, dim, cell, activate = 'ReLU', layerNum = 0.5):
        self.layerNum = layerNum
        self.dim = dim
        self.cell = cell
        self.weight = np.random.rand(cell ,dim +1 ) /2 -0.25  ## (10,785)
        self.activate = activate

    def getWeight(self):
        del_col = self.weight.shape[1] - 1
        return np.delete(self.weight, del_col, 1)

    def getOutput(self):
        return self.output

    def forward(self, X):  ## (784,100)
        self.input = self.appendInput(X)  ## (785,100)
        matrix = self.weight.dot(self.input)
        matrix = matrix.T

        if self.activate == 'ReLU':
            active_matrix = np.array([ReLU(row) for row in matrix])
        else:
            active_matrix = np.array([softmax(row) for row in matrix])
        self.output = active_matrix.T
        return self.output

    def appendInput(self, X):
        dim_y = X.shape[1]
        Y = np.random.rand(1, dim_y) - 0.5
        X = np.append(X, Y, 0)
        return X

    def backward(self, delta, weight, learn):
        ## delta (1,100) weight (1,10)  learn 0.1  output (10,100) input (785,100)
        ## self.delta = sigmoid_derivate(output) * (delta.T dot weight) (10,100)
        ## delta_weight = learn * (input dot self.delta)
        ## self.weight = self.weight - delta_weight (10,785)
        output = self.output
        if self.activate == 'ReLU':
            output_derivate = np.array([ReLU_deriv(x) for row in output for x in row])
        else:
            output_derivate = output
        output_derivate = np.reshape(output_derivate, output.shape)  ##(10,100)
        self.delta = output_derivate * weight.T.dot(delta)  ##(10,100)
        batch_size = delta.shape[1]
        delta_weight = self.delta.dot(self.input.T) / batch_size
        self.weight = self.weight + learn * delta_weight
        return self.delta


class Model:
    def __init__(self, learn):
        self.learn = learn
        self.layers = []
        self

    def addLayer(self, layer):
        self.layers.append(layer)

    def getLayer(self, index):
        size = len(self.layers)
        if index >= size:
            raise Exception("index out of range")
        return self.layers[index]

    def getLayers(self):
        return self.layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        self.output = X
        return X

    def backward(self, delta):
        size = delta.shape[0]
        weight = np.diag([1] * size)
        learn = self.learn
        layers = list(reversed(self.layers))
        for layer in layers:
            delta = layer.backward(delta, weight, learn)
            weight = layer.getWeight()
        return