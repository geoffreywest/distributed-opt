from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from tqdm import trange
import numpy as np
import random
import json
import os
np.random.seed(100)

def gen_orig():
    N_CLIENTS = 100

    # Setup directory for train/test data
    train_path = 'orig/train.json'
    test_path = 'orig/test.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Fetch data and normalize
    mnist = fetch_openml('mnist_784')
    mnist.data = StandardScaler().fit_transform(mnist.data)

    # Create data structure
    train_data = {'users':[], 'user_data':{}, 'num_samples':[]}
    test_data = {'users':[], 'user_data':{}, 'num_samples':[]}

    #  Shuffle data
    perm = np.random.permutation(len(mnist.data))
    mnist.data, mnist.target = mnist.data[perm], mnist.target[perm]
    samples_per_client = len(mnist.data) // N_CLIENTS

    # Setup 1000 users with train/test data each
    for i in trange(N_CLIENTS):

        s = slice(i*samples_per_client, (i+1)*samples_per_client)
        X, y = mnist.data[s], mnist.target[s]

        n_samples = samples_per_client
        n_train = int(.9 * n_samples)
        n_test = n_samples - n_train

        train_data['user_data'][i] = {'X': X[:n_train].tolist(), 'y': y[:n_train].tolist()}
        train_data['num_samples'].append(n_train)
        train_data['users'].append(i)
        test_data['user_data'][i] = {'X': X[n_train:].tolist(), 'y': y[n_train:].tolist()}
        test_data['num_samples'].append(n_test)
        test_data['users'].append(i)

    print('Total samples: {}'.format(sum(train_data['num_samples'])))

    # Write the data to .json
    with open(train_path,'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)

def gen_grouped():
    N_CLIENTS = 100

    # Setup directory for train/test data
    train_path = 'grouped/train.json'
    test_path = 'grouped/test.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Get MNIST data, normalize, and divide by level
    mnist = fetch_openml('mnist_784')
    mnist.data = StandardScaler().fit_transform(mnist.data)
    mnist_data = []

    # Create data structure
    train_data = {'users':[], 'user_data':{}, 'num_samples':[]}
    test_data = {'users':[], 'user_data':{}, 'num_samples':[]}

    #  Shuffle data
    perm = np.random.permutation(len(mnist.data))
    mnist.data, mnist.target = mnist.data[perm], mnist.target[perm].astype(np.float32)

    # Group by target label
    for i in trange(10):
        idx = np.where(mnist.target==i)[0]
        mnist_data.append(idx)

    # Setup 1000 users with train/test data each
    samples_per_client = 600
    for i in trange(N_CLIENTS):

        label1, label2 = i % 10, (i+1) % 10
        idx = np.concatenate([mnist_data[label1][-300:], mnist_data[label2][-300:]])
        mnist_data[label1] = mnist_data[label1][:-300]
        mnist_data[label2] = mnist_data[label2][:-300]
        s = np.random.permutation(idx)
        X, y = mnist.data[s], mnist.target[s]

        n_samples = samples_per_client
        n_train = int(.9 * n_samples)
        n_test = n_samples - n_train

        train_data['user_data'][i] = {'X': X[:n_train].tolist(), 'y': y[:n_train].tolist()}
        train_data['num_samples'].append(n_train)
        train_data['users'].append(i)
        test_data['user_data'][i] = {'X': X[n_train:].tolist(), 'y': y[n_train:].tolist()}
        test_data['num_samples'].append(n_test)
        test_data['users'].append(i)
    print('Total samples: {}'.format(sum(train_data['num_samples'])))

    # Write the data to .json
    with open(train_path,'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)

if __name__ == '__main__':
    gen_grouped()
