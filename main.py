import numpy as np
import torch
import argparse
import importlib
import random
import os
from utils.model_utils import read_data

'''
Modeled on:
https://github.com/litian96/FedProx/blob/master/main.py
'''

# GLOBAL PARAMETERS
OPTIMIZERS = ['fedavg', 'fedprox', 'feddane', 'fedddane', 'fedsgd', 'fedprox_origin', 'fedio', 'fedpdsvrg']
DATASETS = ['sent140', 'nist', 'shakespeare', 'mnist',
'synthetic_iid', 'synthetic_0_0', 'synthetic_0.5_0.5', 'synthetic_1_1']  # NIST is EMNIST in the paepr TODO


MODEL_PARAMS = {
    'sent140.bag_dnn': (2,), # num_classes
    'sent140.stacked_lstm': (25, 2, 100), # seq_len, num_classes, num_hidden
    'sent140.stacked_lstm_no_embeddings': (25, 2, 100), # seq_len, num_classes, num_hidden
    'nist.mclr': (26,),  # num_classes
    'mnist.mclr': (10,), # num_classes
    'mnist.cnn': (10,),  # num_classes
    'shakespeare.stacked_lstm': (80, 80, 256), # seq_len, emb_dim, num_hidden
    'synthetic.mclr': (10,) # num_classes
}


def read_options():
    '''
    Read options from the command line.

    Return:
        1. parsed: a dict containing
            - all the command line arguments
            - the relevant MODEL_PARAMS tuple as listed above
        2. model: a model class
        3. server: a server type
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer',
                        help='name of optimizer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='fedavg')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        choices=DATASETS,
                        default='nist')
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='stacked_lstm.py')
    parser.add_argument('--num_rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=-1)
    parser.add_argument('--eval_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=-1)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=-1)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=10)
    parser.add_argument('--num_epochs',
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=1)
    parser.add_argument('--num_iters',
                        help='number of iterations when clients train on data;',
                        type=int,
                        default=1)
    parser.add_argument('--learning_rate',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.003)
    parser.add_argument('--rho',
                        help='inner step size proportion for inner outer method',
                        type=float,
                        default=1.0)
    parser.add_argument('--mu',
                        help='constant for prox;',
                        type=float,
                        default=0)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parser.add_argument('--drop_percent',
                        help='percentage of slow devices',
                        type=float,
                        default=0.1)

    try: parsed = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))

    # Set seeds
    random.seed(1 + parsed['seed'])
    np.random.seed(12 + parsed['seed'])
    torch.manual_seed(123 + parsed['seed'])

    # Load model
    # TODO do we need the special case
    if parsed['dataset'].startswith('synthetic'):
        model_path = '%s.%s.%s' % ('models', 'synthetic', parsed['model'])
    else:
        model_path = '%s.%s.%s' % ('models', parsed['dataset'], parsed['model'])

    mod = importlib.import_module(model_path)
    model = getattr(mod, 'Model')

    # Load trainer
    opt_path = 'trainers.%s' % parsed['optimizer']
    mod = importlib.import_module(opt_path)
    server = getattr(mod, 'Server')

    # Add selected model parameter
    parsed['model_params'] = MODEL_PARAMS['.'.join(model_path.split('.')[1:])]

    # Print arguments
    max_len = max([len(ii) for ii in parsed.keys()])
    fmt_string = '\t%' + str(max_len) + 's : %s'
    print('Arguments:')
    for key_val in sorted(parsed.items()): print(fmt_string % key_val)

    return parsed, model, server



def main():

    # Parse command line arguments
    options, model, server = read_options()

    # Read data
    train_path = os.path.join('data', options['dataset'], 'orig', 'train.json')
    test_path = os.path.join('data', options['dataset'], 'orig', 'test.json')
    print('Reading data...')
    dataset = read_data(train_path, test_path)
    print('Reading data done.')

    optim = server(options, model, dataset)
    optim.run()
    optim.save()

if __name__ == '__main__':
    main()
