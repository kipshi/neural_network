import numpy as np
import math

def softmax(Z):
    A = np.exp2(Z) / sum(np.exp2(Z))
    return A

def sigmoid(x):
    return 1.0/(1.0+math.exp(-x))

def sigmoid_derivate(x):
    return x*(1-x)

def ReLU(Z):
    A = np.maximum(Z, 0)
    return A

def ReLU_deriv(Z):
    return 1 if Z > 0 else 0

def get_predictions(A2):
    return np.argmax(A2, 0)

def isNaN(Z):
    isnan = np.isnan(Z)
    return True in isnan


def findMax(Z):
    max_num = np.max(Z)
    for row in range(Z.shape[0]):
        for col in range(Z.shape[1]):
            if Z[row,col]== max_num:
                return (row,col)