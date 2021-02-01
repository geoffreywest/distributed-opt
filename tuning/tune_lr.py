import numpy as np
import pandas as pd
import torch
import argparse
import importlib
import itertools
import random
import sys
import os
from tqdm import tqdm
from multiprocessing import Pool

sys.path.append('/Users/geoffreywest/Desktop/Research/Srebro/Code/distributed-opt/')
import trainers.fedio
import trainers.fedpdsvrg
import trainers.fedprox
#from models.mnist.mclr import Model
from utils.model_utils import read_data
from utils.model_utils import Metrics

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

OPTIMIZERS = ['fedavg', 'fedprox', 'feddane', 'fedddane', 'fedsgd', 'fedprox_origin', 'fedio', 'fedpdsvrg']
DATASETS = ['sent140', 'nist', 'shakespeare', 'mnist',
'synthetic_iid', 'synthetic_0_0', 'synthetic_0.5_0.5', 'synthetic_1_1']  # NIST is EMNIST in the paper TODO
DATALOCS = ['orig', 'grouped']

def callback(lr, seed, dataset, options, model_path, server_path):
    options['learning_rate'] = lr
    options['seed'] = seed
    options['verbosity'] = 0
    #options['optimizer'] = 'fedio'

    # Set seeds
    random.seed(1 + seed)
    np.random.seed(12 + seed)
    torch.manual_seed(123 + seed)

    model_mod = importlib.import_module(model_path)
    server_mod = importlib.import_module(server_path)

    model = getattr(model_mod, 'Model')
    server = getattr(server_mod, 'Server')
    optimizer = server(options, model, dataset)
    optimizer.run()
    print('Callback done.')
    return optimizer.metrics

def get_search_space(mid, coef, pow_low, pow_high, **kwargs):
    return mid * ((coef * np.ones(pow_high - pow_low)) ** np.arange(pow_low, pow_high))

def read_options():

    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer',
                        help='name of optimizer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='fedio')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        choices=DATASETS,
                        default='mnist')
    parser.add_argument('--dataloc',
                        help='name of dataset;',
                        type=str,
                        choices=DATALOCS,
                        default='orig')
    parser.add_argument('--model',
                        help='which model to use',
                        type=str,
                        default='mclr')
    parser.add_argument('--num_seeds',
                        help='number of random seeds to tune on',
                        type=int,
                        default=1)
    parser.add_argument('--num_epochs',
                        help='number of epochs to locally optimize each round',
                        type=int,
                        default=1)
    parser.add_argument('--batch_size',
                        help='the batch size in the local optimization',
                        type=int,
                        default=1)
    parser.add_argument('--rho',
                        help='proportion of inner/outer learniing rate',
                        type=float,
                        default=.5)
    parser.add_argument('--mu',
                        help='weight on proximal term',
                        type=float,
                        default=0)
    parser.add_argument('--mid',
                        help='median in the learning rate search',
                        type=float,
                        default=.1)
    parser.add_argument('--coef',
                        help='proportion of successive values in the learning rate search',
                        type=float,
                        default=1.2)
    parser.add_argument('--pow_low',
                        help='lowest power in the learning rate search',
                        type=int,
                        default=-5)
    parser.add_argument('--pow_high',
                        help='highest power in the learning rate search',
                        type=int,
                        default=5)
    parser.add_argument('--clients_per_round',
                        help='number of machines each round',
                        type=int,
                        default=10)
    parser.add_argument('--num_rounds',
                        help='number of rounds of communication',
                        type=int,
                        default=20)
    parser.add_argument('--num_iters',
                        help='number of iterations per machine per round',
                        type=int,
                        default=10)
    parser.add_argument('--seed',
                        help='random seed',
                        type=int,
                        default=100)
    parser.add_argument('--drop_percent',
                        help='percentage of slow devices',
                        type=float,
                        default=0)
    parser.add_argument('--eval_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=-1)
    parser.add_argument('--note',
                        help='optional special identifier to put in the results file name',
                        type=str,
                        default='')

    try: parsed = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))

    parsed['model_params'] = MODEL_PARAMS['{}.{}'.format(parsed['dataset'], parsed['model'])]

    return parsed

def parallel_execute(args, dataset, options, model_path, server_path):
    '''
    Input:
        - args: a list of (lr, seed) pairs to experiment
        - options: dict of command line options
    Returns:
        - list of metrics objects from each run
    '''
    with Pool(5) as p:
        args_packed = [(arg[0], arg[1], dataset, options, model_path, server_path) for arg in args]
        res = tqdm(p.starmap(callback, args_packed)).iterable
    return res


def reduce_results(args, results, getter_fn=Metrics.mean_loss, reducer_fn=np.mean):
    '''
    Return the mean loss for each
    '''
    vals = map(getter_fn, results)
    grouped = {}
    for (lr, seed), res in zip(args, vals):
        if lr not in grouped.keys():
            grouped[lr] = []
        grouped[lr].append(res)
    lrs = sorted(list(set([lr for lr, _ in args])))
    return [reducer_fn(grouped[lr]) for lr in lrs]



def main():
    # Read in the command line options
    options = read_options()
    # Generate the search space for learning rates
    search = get_search_space(**options)
    # Generate the random seeds to test
    seeds = range(options['seed'], options['seed'] + options['num_seeds'])
    # Read in the data
    train_path = os.path.join('..','data', options['dataset'], options['dataloc'], 'train.json')
    test_path = os.path.join('..','data', options['dataset'], options['dataloc'], 'test.json')
    print('Reading data...')
    dataset = read_data(train_path, test_path)
    print('Reading data done.')
    # Import model
    if options['dataset'].startswith('synthetic'):
        model_path = '%s.%s.%s' % ('models', 'synthetic', options['model'])
    else:
        model_path = '%s.%s.%s' % ('models', options['dataset'], options['model'])
    # Import server
    server_path = 'trainers.{}'.format(options['optimizer'])
    # Try all seeds and learning rates
    args = list(itertools.product(search,seeds))
    results = parallel_execute(args, dataset, options, model_path, server_path)
    losses = reduce_results(args, results)
    results_reduced = list(zip(search,losses))
    print(results_reduced)
    print('Best result: {} (index {})'.format(results_reduced[np.argmin(losses)], np.argmin(losses)))
    # Make output directory
    if not os.path.exists('results'):
        os.mkdir('results')
    note = '_' + options['note'] if len(options['note']) > 0 else ''
    def get_aux_param(options):
        optim = options['optimizer']
        if optim == 'fedio':
            return options['rho']
        elif optim == 'fedprox':
            return options['mu']
        return 0
    out_file = 'results/tune_{}_{}_{}_{}_{:.3f}_{}_{}_{}_{}_{}{}.csv'.format(
        options['optimizer'],
        options['dataset'],
        options['dataloc'],
        options['model'],
        get_aux_param(options),
        options['num_rounds'],
        options['clients_per_round'],
        options['num_iters'],
        options['seed'],
        options['num_seeds'],
        note
    )
    pd.DataFrame(results_reduced, columns=['lr', 'loss']).set_index('lr').to_csv(out_file)
    return

if __name__  == '__main__':
    main()
