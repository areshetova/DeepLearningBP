import os
import gzip
import sys
import math
import random
import numpy as np

from urllib.request import urlretrieve

def download(filename, source, target_file):
    urlretrieve(source + filename, target_file)

def download_images(filename):
    with gzip.open(filename, 'rb') as file:
        data = np.frombuffer(file.read(), np.uint8, offset=16)
        img_data = data.reshape(-1, 784)
    return img_data

def download_labels(filename):
    with gzip.open(filename, 'rb') as file:
        data = np.frombuffer(file.read(), np.uint8, offset=8)
    return data

def download_images_label_pair(images_file, labels_file):
    images_file_on_disk = "./Data_folder/" + images_file
    if not os.path.exists(images_file_on_disk):
        download(images_file, "http://yann.lecun.com/exdb/mnist/", images_file_on_disk)
    images = download_images(images_file_on_disk)
    labels_file_on_disk = "./Data_folder/" + labels_file
    if not os.path.exists(labels_file_on_disk):
        download(labels_file, "http://yann.lecun.com/exdb/mnist/", labels_file_on_disk)
    labels = download_labels(labels_file_on_disk)
    return images, labels

def retrieve_sample_data():
    if not os.path.exists("./Data_folder/"):
        os.makedirs("./Data_folder/")
    train_imgs, train_lbls = download_images_label_pair("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")
    test_imgs, test_lbls = download_images_label_pair("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
    return train_imgs, train_lbls, test_imgs, test_lbls

def normalize(image: np.array):
    return image / 255.0

def convert_number_to_array(y_list, classes):
    result = []
    for value in y_list:
        array_from_value = np.zeros(classes)
        array_from_value[value] = 1
        result.append(array_from_value)
    return np.array(result)

def main():
    args = sys.argv
    
    np.random.seed(13)
    if '-s' in args:
        idx = args.index('-s')
        neuron_count = int(args[idx+1])
    else:
        neuron_count = 100
    y_classes = 10
    range_of_weights = (0.003, 0.007)
    if '-l' in args:
        idx = args.index('-l')
        train_speed = float(args[idx+1])
    else:
        train_speed = 0.05
    if '-e' in args:
        idx = args.index('-e')
        max_epoch = int(args[idx+1])
    else:
        max_epoch = 25
    if '-a' in args:
        idx = args.index('-a')
        max_accuracy = float(args[idx+1])
    else:
        max_accuracy = 0.99

    def accuracy(X_array: np.array, y_array: np.array, weightes_hidden_layer :np.array, weightes_output_layer :np.array):
        true_classifications = 0
        for x, y in zip(X_array, y_array):
            label = np.argmax(y)
            hidden_level = sigmoid(np.dot(weightes_hidden_layer, x))
            output_level = softmax(np.dot(weightes_output_layer, hidden_level))
            prediction = np.argmax(output_level)
            if (prediction == label):
                true_classifications += 1
        return true_classifications / X_array.shape[0]

    def Cross_entropy_error(y_array: np.array, t_array: np.array):
        sumError = 0.0
        for i in range(len(t_array)):
            sumError += math.log(y_array[i]) * t_array[j]
        return -1.0 * sumError
  
    def sigmoid(x):
        return 1./(1. + np.exp(-x))

    def sigmoid_grad(x):
        return x * (1. - x)

    def softmax(x):
        max_value = np.max(x)
        shifted_args = x - max_value
        exps = np.exp(shifted_args)
        return exps / np.sum(exps)

    def softmax_grad(x):
        return x * (1. - x)

    def initialize_weights(size: np.array, val_range: np.array):
        return val_range[0] + (val_range[1]-val_range[0])*np.random.random((size[0], size[1]))
 
    print("Downloading")
    train_data, train_labels, test_data, test_labels = retrieve_sample_data()
    X_train = np.array([normalize(np.array(image)) for image in train_data])
    y_train = convert_number_to_array(train_labels, y_classes)
    X_test = np.array([normalize(np.array(image)) for image in test_data])
    y_test = convert_number_to_array(test_labels, y_classes)
    
    x_params_count = X_train.shape[1]
    weightes_hidden_layer = initialize_weights((neuron_count, x_params_count), range_of_weights)    
    weightes_output_layer = initialize_weights((y_classes, neuron_count), range_of_weights)

    print("Training:")
    for epoch in range(max_epoch):
        for x, y in zip(X_train, y_train):
            # forward propagation
            weighted_input_hidden_layer = np.dot(weightes_hidden_layer, x)
            hidden_layer_output = sigmoid(weighted_input_hidden_layer)
            
            weighted_input_output_layer = np.dot(weightes_output_layer, hidden_layer_output)
            final_layer_output = softmax(weighted_input_output_layer)
            
            # backward propagation
            final_level_error = final_layer_output - y
            final_level_transferred_error = weightes_output_layer.T.dot(final_level_error)
            
            hidden_level_error = final_level_transferred_error * softmax_grad(hidden_layer_output)
            
            # update weights
            weightes_output_layer -= train_speed * np.outer(final_level_error, hidden_layer_output)
            
            weightes_hidden_layer -= train_speed * np.outer(hidden_level_error, x)            
        
        current_accuracy = accuracy(X_train, y_train, weightes_hidden_layer, weightes_output_layer)
        
        print("  Epoch {0}: accuracy = {1}".format(epoch+1, current_accuracy))
        if (current_accuracy > max_accuracy):
            print("Maximum accuracy achieved")
            break

    print("Testing:")
    result_accuracy = accuracy(X_test, y_test, weightes_hidden_layer, weightes_output_layer)
    print("Test accuracy = {0}".format(result_accuracy))

if __name__ == '__main__':
    main()
