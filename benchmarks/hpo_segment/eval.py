import sys
import os
from os.path import join, dirname, abspath
import numpy as np
import ConfigSpace
import warnings
import argparse
sys.path.append('hypermapper')
import optimizer as bopro
warnings.filterwarnings('ignore')
from ConfigSpace import Configuration

sys.path.append(join(dirname(dirname(abspath(__file__))),"libs/HPOBench"))
from hpobench.benchmarks.ml.nn_benchmark import NNBenchmark as Benchmark


def format_arguments(alpha, batch_size, depth, learning_rate_init, width):
    alpha = np.power(10, alpha).astype(float)
    batch_size = np.round(np.power(2, batch_size)).astype(int)
    depth = int(depth) + 1
    learning_rate_init = np.power(10, learning_rate_init).astype(float)
    width = np.round(np.power(2, width)).astype(int)
    args = {
        'alpha': alpha,
        'batch_size': batch_size,
        'depth': depth,
        'learning_rate_init': learning_rate_init, 
        'width': width
    }
    return args


def eval_function(args):
    args_dict = format_arguments(**args)

    b = Benchmark(task_id=146822, rng=1)
    config = Configuration(b.get_configuration_space(seed=1), args_dict)

    result_dict = b.objective_function(configuration=config, rng=1)
    return result_dict['function_value']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', type=int, help='Index of the prior.', default=0)
    args = parser.parse_args()
    index = args.index
    dirname = dirname(abspath(__file__)).split('/')[-1]
    parameters_file = f"benchmarks/{dirname}/{dirname}_scenario.json"
    bopro.optimize(parameters_file, eval_function, output_file=f'results/{dirname}_run_{index}.csv')
    print("End of HPO.")

