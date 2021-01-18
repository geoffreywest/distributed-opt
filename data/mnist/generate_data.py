from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from tqdm import trange
import numpy as np
import random
import json
import os
np.random.seed(100)

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
