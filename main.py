from matplotlib import pyplot as plt
from model import *

import tensorflow as tf
mnist = tf.keras.datasets.mnist

def one_hot(y):
    one_hot_y = np.zeros((y.max()+1,y.size))
    one_hot_y[y,np.arange(y.size)] = 1
    return one_hot_y

def get_predicts(Z):
    return np.argmax(Z,0)

def get_accuracy(y,predict):
    return np.sum(predict == y)/y.size

def print_image(Z):
        plt.gray()
        plt.imshow(Z, interpolation='nearest')
        plt.show()


def modify_input(input):
    shape = input.shape
    if len(shape) == 3:
        size = input.shape[1] * input.shape[2]
        batch_size = input.shape[0]
    else:
        size = input.shape[0] * input.shape[1]
        batch_size = 1

    input = np.reshape(input, (batch_size, size)).T
    return input / np.max(input)

if __name__ == '__main__':

    ## load data from mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    ## create model
    model = Model(0.05)
    layer1 = Layer(784, 100, layerNum=1)
    model.addLayer(layer1)
    layer2 = Layer(100, 50, layerNum=2)
    model.addLayer(layer2)
    layer3 = Layer(50, 10, 'softmax', layerNum=3)
    model.addLayer(layer3)

    ## train model
    for time in range(30):
        accuracy = 0

        for index in range(int(x_train.shape[0] / 100)):
            start = index * 100
            end = (index + 1) * 100
            x = x_train[start: end]
            x = modify_input(x)
            predict = model.forward(x)
            y = y_train[start: end]
            y = one_hot(y)
            layer1 = model.getLayer(0)
            layer2 = model.getLayer(1)
            delta = y - predict
            model.backward(delta)

            y = get_predicts(y)
            predict = get_predicts(predict)
            new_accuracy = get_accuracy(y, predict)
            accuracy = new_accuracy if new_accuracy > accuracy else accuracy
            print('accuracy', accuracy)

        if (accuracy >= 0.9):
            break

    ## check predict for index
    print(x_test.shape)
    index = 0

    x = x_test[index]
    y = y_test[index]
    print_image(x)
    print(y)
    x = modify_input(x)
    predict = get_predicts(model.forward(x))
    print(predict)

    ## check batch_size predict accuracy
    batch_size = 10000

    x_batch = x_test[index:index + batch_size]
    x_batch = modify_input(x_batch)
    predict_batch = get_predicts(model.forward(x_batch))
    y_batch = y_test[index:index + batch_size]

    accuracy = get_accuracy(y_batch, predict_batch)
    print(accuracy)
