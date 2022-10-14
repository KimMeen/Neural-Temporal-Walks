import numpy as np
import torch
import os
import random
import argparse
import sys


def get_args():
    parser = argparse.ArgumentParser('Neural Temporal Walks (NeurIPS 2022)')

    # General
    parser.add_argument('-d', '--data', type=str, choices=['CollegeMsg', 'enron', 'TaobaoSmall', 'mooc',
                                                           'wikipedia', 'reddit'],
                        default='CollegeMsg',
                        help='dataset to use')
    parser.add_argument('--data_usage', default=1.0, type=float,
                        help='fraction of data to use (0-1)')
    parser.add_argument('-m', '--mode', type=str, default='t', choices=['t', 'i'],
                        help='transductive (t) or inductive (i)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='the GPU to be used')
    parser.add_argument('--cpu_cores', type=int, default=1,
                        help='number of cpu_cores used for position encoding')

    # Training-related
    parser.add_argument('--n_epoch', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('--bs', type=int, default=32,
                        help='training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--drop_out', type=float, default=0.1,
                        help='dropout probability for all dropout layers')
    parser.add_argument('--tolerance', type=float, default=0,
                        help='tolerated marginal improvement for early stopper')

    # Model-related
    parser.add_argument('--n_degree', nargs='*', default=['64', '1'],
                        help='a list of neighbor sampling numbers for different hops, '
                             'when only a single element is input n_layer will be activated')
    parser.add_argument('--n_layer', type=int, default=2,
                        help='number of layers to be sampled (only valid when n_degree has a single element)')
    parser.add_argument('--pos_enc', type=str, default='saw', choices=['saw', 'lp'],
                        help='unitary or binary position encoding')
    parser.add_argument('--pos_dim', type=int, default=172,
                        help='dimension of the positional encoding')
    parser.add_argument('--pos_sample', type=str, default='multinomial', choices=['multinomial'],
                        help='spatiotemporal-biased walk sampling')
    parser.add_argument('--walk_mutual', action='store_true',
                        help="whether to do mutual query for source and target node random walks")
    parser.add_argument('--walk_linear_out', action='store_true', default=False,
                        help="whether to linearly project each node's embedding")
    parser.add_argument('--temporal_bias', default=1e-5, type=float,
                        help='temporal-bias intensity')
    parser.add_argument('--spatial_bias', default=1.0, type=float,
                        help='spatial-bias intensity')
    parser.add_argument('--ee_bias', default=1.0, type=float,
                        help='ee-bias intensity')
    parser.add_argument('--solver', type=str, default='rk4', choices=['euler', 'rk4', 'dopri5'],
                        help='the ODE solver to be used')
    parser.add_argument('--step_size', type=float, default=0.125,
                        help='step size to be used in fixed-step solvers (e.g., euler and rk4)')
    parser.add_argument('--tau', type=float, default=0.1,
                        help='temperature in noise-contrastive loss')
    parser.add_argument('--negs', type=int, default=1,
                        help='number of negatives in noise-contrastive loss')
    parser.add_argument('--limit_ngh_span', action='store_true',
                        help="whether to limit the maximum number of spanned temporal neighbors")
    parser.add_argument('--ngh_span', nargs='*', default=['320', '8'],
                        help='a list of maximum number of spanned temporal neighbors for different hops')
    parser.add_argument('--ngh_cache', action='store_true',
                        help='(currently not suggested due to overwhelming memory consumption)'
                             'cache temporal neighbors previously calculated to speed up repeated lookup')
    parser.add_argument('--verbosity', type=int, default=1,
                        help='verbosity of the program output')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    return args, sys.argv


class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        self.epoch_count += 1
        return self.num_round >= self.max_round


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list):
        src_list = np.concatenate(src_list)
        dst_list = np.concatenate(dst_list)
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

    def sample(self, size):
        src_index = np.random.randint(0, len(self.src_list), size)
        dst_index = np.random.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def process_sampling_numbers(num_neighbors, num_layers):
    num_neighbors = [int(n) for n in num_neighbors]
    if len(num_neighbors) == 1:
        num_neighbors = num_neighbors * num_layers
    else:
        num_layers = len(num_neighbors)
    return num_neighbors, num_layers
