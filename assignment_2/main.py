import argparse
import os
import logging

import numpy

from multilayer_perceptron import MultilayerPerceptron

data_dir = os.path.join(os.path.dirname(__file__), 'data')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Program arguments
    parser.add_argument('-v', help='Output verbosity, add more v to increase', dest='verbosity', action='count',
                        default=0)
    parser.add_argument('--mnist_range', help='The range of MNIST numbers to train on', type=int, nargs=2,
                        default=(0, 1))
    # Network arguments
    parser.add_argument('-e', '--epochs', help='Number of epochs to train for', type=int, default=1)
    args = parser.parse_args()

    # Set up logging
    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(args.verbosity, len(levels) - 1)]  # cap to last level index
    logging.basicConfig(format='%(levelname)s:%(message)s', level=level)

    training_data_file = os.path.join(data_dir, f'mnist_train_{args.mnist_range[0]}_{args.mnist_range[1]}.csv')
    testing_data_file = os.path.join(data_dir, f'mnist_test_{args.mnist_range[0]}_{args.mnist_range[1]}.csv')

    # check if the data files exist
    if not os.path.exists(training_data_file):
        logging.error(f'Training data file at "{training_data_file}" does not exist')
        raise FileNotFoundError(f'Training data file at "{training_data_file}" does not exist')
    if not os.path.exists(testing_data_file):
        logging.error(f'Testing data file at "{testing_data_file}" does not exist')
        raise FileNotFoundError(f'Testing data file at "{testing_data_file}" does not exist')

    # read in the data
    training_data = numpy.loadtxt(training_data_file, delimiter=',')
    testing_data = numpy.loadtxt(testing_data_file, delimiter=',')

    mlp = MultilayerPerceptron(num_features=784, hidden_nodes=4, output_nodes=1)

    for epoch in range(4):
        logging.info(f'Epoch {epoch}')
        mlp.train(training_data, 0.05)
        logging.info(f'Train Accuracy: {mlp.test(training_data)} Test Accuracy: {mlp.test(testing_data)}')

    logging.info('Training complete')
    logging.info(f'Final Accuracy: {mlp.test(testing_data)}')
