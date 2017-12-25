import os
import gzip
import sys
import math
import random
import numpy as np

from urllib.request import urlretrieve

import NeuralNetwork as nn

def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
    print("Downloading %s" % filename)
    urlretrieve(source + filename, filename)
        
def download_images(filename):
    if not os.path.exists(filename):
        download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 784)
    return data / np.float32(256)

def download_labels(filename):
    if not os.path.exists(filename):
        download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    n = len(data)
    no = max(data)
    labels = np.zeros(shape=[n,no+1], dtype=np.float32)
    for i in range(n):
        vec = np.zeros(shape=[no+1], dtype=np.float32)
        vec[data[i]] = 1
        labels[i] = vec
    return labels

def main():
    args = sys.argv

    if '-n' in args:
        idx = args.index('-n')
        train_im = int(args[idx+1])
    else:
        train_im = 1000
    if '-t' in args:
        idx = args.index('-t')
        test_im = int(args[idx+1])
    else:
        test_im = 100
    if '-s' in args:
        idx = args.index('-s')
        hidden_neurons = int(args[idx+1])
    else:
        hidden_neurons = 100
    if '-l' in args:
        idx = args.index('-l')
        train_speed = float(args[idx+1])
    else:
        train_speed = 0.005

    input_neurons = 28 * 28
    output_neurons = 10
    max_epochs = 15
    cross_error = 0.005

    X_train = download_images('train-images-idx3-ubyte.gz')
    t_train = download_labels('train-labels-idx1-ubyte.gz')
    X_test = download_images('t10k-images-idx3-ubyte.gz')
    t_test = download_labels('t10k-labels-idx1-ubyte.gz')

    X_train = X_train[0:train_im]
    t_train = t_train[0:train_im]

    X_test = X_test[0:test_im]
    t_test = t_test[0:test_im]

    inputNodes = np.zeros(shape=[input_neurons], dtype=np.float32)
    hiddenNodes = np.zeros(shape=[hidden_neurons], dtype=np.float32)
    outputNodes = np.zeros(shape=[output_neurons], dtype=np.float32)

    input_hidden_weights = np.zeros(shape=[input_neurons,hidden_neurons], dtype=np.float32)
    hidden_output_weights = np.zeros(shape=[hidden_neurons,output_neurons], dtype=np.float32)

    hBiases = np.zeros(shape=[hidden_neurons], dtype=np.float32)
    oBiases = np.zeros(shape=[output_neurons], dtype=np.float32)

    rnd = random.Random(10)

    for i in range(input_neurons):
        for j in range(hidden_neurons):
            input_hidden_weights[i,j] = 0.5 * rnd.random()
    for i in range(hidden_neurons):
        for j in range(output_neurons):
            hidden_output_weights[i,j] = 0.5 * rnd.random()
    for i in range(hidden_neurons):
        hBiases[i] = 0.5 * rnd.random()
    for i in range(output_neurons):
        oBiases[i] = 0.5 * rnd.random()

    def computeOutputs(xValues):
        hSums = np.zeros(shape=[hidden_neurons], dtype=np.float32)
        oSums = np.zeros(shape=[output_neurons], dtype=np.float32)

        for i in range(input_neurons):
            inputNodes[i] = xValues[i]

        for i in range(hidden_neurons):
            for j in range(input_neurons):
                hSums[i] += inputNodes[j] * input_hidden_weights[j,i]
            hSums[i] += hBiases[i]

        for i in range(hidden_neurons):
            hiddenNodes[i] = hypertan(hSums[i])

        for i in range(output_neurons):
            for j in range(hidden_neurons):
                oSums[i] += hiddenNodes[j] * hidden_output_weights[j,i]
            oSums[i] += oBiases[i]

        softOut = softmax(oSums)
        for i in range(output_neurons):
            outputNodes[i] = softOut[i]
        return outputNodes

    def computeGradient(t_values, oGrads, hGrads):
        for i in range(output_neurons):
            oGrads[i] = (t_values[i] - outputNodes[i])

        for i in range(hidden_neurons):
            derivative = (1 - hiddenNodes[i]) * (1 + hiddenNodes[i]);
            sum_ = 0.0
            for j in range(output_neurons):
                sum_ += oGrads[j] * hidden_output_weights[i, j]
            hGrads[i] = derivative * sum_

    def updateWeightsAndBiases(learnRate, hGrads, oGrads):
        for i in range(input_neurons):
            for j in range(hidden_neurons):
                input_hidden_weights[i,j] += learnRate * hGrads[j] * inputNodes[i]
        for i in range(hidden_neurons):
            for j in range(output_neurons):
                hidden_output_weights[i,j] += learnRate * oGrads[j] * hiddenNodes[i];

        for i in range(hidden_neurons):
            hBiases[i] += learnRate * hGrads[i] * 1.0
        for i in range(output_neurons):
            oBiases[i] += learnRate * oGrads[i] * 1.0

    def crossEntropyError(x_values, t_values):
        sumError = 0.0
        for i in range(len(x_values)):
            y_values = computeOutputs(x_values[i])
            for j in range(output_neurons):
                sumError += math.log(y_values[j]) * t_values[i][j]
        return -1.0 * sumError / len(x_values)

    def accuracy(x_values, t_values):
        num_correct = 0
        num_wrong = 0
        for i in range(len(x_values)):
            y_values = computeOutputs(x_values[i])
            max_index = np.argmax(y_values)
            if abs(t_values[i][max_index] - 1.0) < 1.0e-5:
                num_correct += 1
            else:
                num_wrong += 1
        return (num_correct * 1.0) / (num_correct + num_wrong)*100

    def train(x_values, t_values, maxEpochs, learnRate, crossError):
        hGrads = np.zeros(shape=[hidden_neurons], dtype=np.float32)
        oGrads = np.zeros(shape=[output_neurons], dtype=np.float32)
        rand_indicies = np.arange(len(x_values))
        for epoch in range(maxEpochs):
            print("Epoch ", epoch)
            np.random.shuffle(rand_indicies)
            x_values = x_values[rand_indicies]
            t_values = t_values[rand_indicies]

            for i in range(len(x_values)):
                outputNodes = computeOutputs(x_values[i])
                computeGradient(t_values[i], oGrads, hGrads)
                updateWeightsAndBiases(learnRate, hGrads, oGrads)
            currentCrossError = crossEntropyError(x_values, t_values)
            if (currentCrossError < crossError):
                return;
            
    def hypertan(x):
        if x < -20.0:
            return -1.0
        elif x > 20.0:
            return 1.0
        else:
            return math.tanh(x)

    def softmax(oSums):
        result = np.zeros(shape=[len(oSums)], dtype=np.float32)
        max_ = max(oSums)
        divisor = 0.0
        for elem in oSums:
            divisor += math.exp(elem - max_)
        for i,elem in enumerate(oSums):
            result[i] =  math.exp(elem - max_) / divisor
        return result
	
    train(X_train, t_train, max_epochs, train_speed, cross_error)

    print("Train: ", accuracy(X_train, t_train), "% Test:", accuracy(X_test, t_test), "%")

if __name__ == '__main__':
    main()
