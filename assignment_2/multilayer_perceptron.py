import logging
import math
import numpy
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

"""
Assignment:
Each training example consists of a comma separated list of 785 values. The first element is the numeric label, and the
 remaining 784 elements are values for each pixel in a 28x28 image of a handwritten character.

Implement a multilayer perceptron to predict the label of images of handwritten 0’s and 1’s. To train this network, you
 will be using backpropagation.

steps to implement:
 0. randomly initialize weights
 1. Generate network output (forward pass)
 2. Calculate output error
 3. Compute delta at the output
 4. Propagate delta to previous layer
 5. Update weights between previous layer and current layer
 6. Repeat steps 4-6 until weights between input and first hidden layer are updated
 
 The matrix_mult_example.py is a partial example of some parts of this assignment which may be helpful.
 The working_example.py is an example using the same dataset but with a different implementation. Don't copy this code.
"""


class MultilayerPerceptron:

    @staticmethod
    def sigmoid(x: numpy.ndarray) -> numpy.ndarray:
        """
        Sigmoid function
        :param x: Input
        :return: Sigmoid of x
        """
        return numpy.tanh(x * 0.5) * 0.5 + 0.5

    @staticmethod
    def sigmoid_derivative(x: numpy.ndarray) -> numpy.ndarray:
        """
        Derivative of the sigmoid function
        :param x: Input
        :return: Derivative of the sigmoid function at x
        """
        return MultilayerPerceptron.sigmoid(x) * (1 - MultilayerPerceptron.sigmoid(x))

    def __init__(self, num_features, hidden_nodes, output_nodes):
        """
        :param num_features: Number of features in the input data
        :param hidden_nodes: Number of "hidden" (middle) nodes in the network, assume one hidden layer
        :param output_nodes: Number of output nodes in the network
        :param bias: Bias value, constant input to the network to allow for a non-zero output when all inputs are zero
        """
        self._num_features = num_features
        self._hidden_nodes = hidden_nodes
        self._output_nodes = output_nodes
        self._biases = [numpy.random.uniform(low=-1, high=1, size=hidden_nodes),
                        numpy.random.uniform(low=-1, high=1, size=output_nodes)]

        # randomly initialize weights as -1,0,1
        self.hidden_weights = numpy.random.uniform(low=-1, high=1, size=(num_features, hidden_nodes))
        self.output_weights = numpy.random.uniform(low=-1, high=1, size=(hidden_nodes, output_nodes))

    @property
    def num_features(self) -> int:
        return self._num_features

    @property
    def hidden_nodes(self) -> int:
        return self._hidden_nodes

    @property
    def output_nodes(self) -> int:
        return self._output_nodes

    def train(self, data: numpy.ndarray, alpha: float):
        """
        Train the network using backpropagation

        :param data: Training data
        :param alpha: Learning rate
        """

        # worth noting that the data is a numpy array of shape (n, 785) where n is the number of training examples

        with logging_redirect_tqdm():
            for row in tqdm(data):
                # logging.debug(f'Input: \n{row}')
                # logging.debug(f'Output Weight Matrix: \n{self.output_weights}')
                # logging.debug(f'Hidden Weight Matrix: \n{self.hidden_weights}')

                input_layer = row[1:]

                # Generate network output (forward pass)
                # hidden layer output
                # OutH = sigmoid((Wh * In) + bn)
                hidden_layer_input = numpy.add(numpy.dot(self.hidden_weights.T, input_layer), self._biases[0])
                hidden_layer_output = MultilayerPerceptron.sigmoid(hidden_layer_input)

                # generate the full output
                # Out = sigmoid((Wo * OutH) + bo)
                output_layer_input = numpy.add(numpy.dot(self.output_weights.T, hidden_layer_output), self._biases[1])
                output_layer_output = MultilayerPerceptron.sigmoid(output_layer_input)

                # delta calculation for output and hidden
                # delta_output = (Y-O) * output_layer_output
                # where Y is the target value and O is the output value
                # delta_output = (row[0] - output_layer_output) * output_layer_output
                delta_output = numpy.multiply(numpy.subtract(row[0], output_layer_output),
                                              MultilayerPerceptron.sigmoid_derivative(output_layer_input))

                # delta_hidden = (Wh * delta_output) * hidden_layer_output
                # where W is the weight and delta_output is the delta value of the output layer
                delta_hidden_dot = numpy.dot(self.output_weights, delta_output)
                delta_hidden_sigmoid_derivative = MultilayerPerceptron.sigmoid_derivative(hidden_layer_input)
                delta_hidden = numpy.multiply(numpy.dot(self.output_weights, delta_output),
                                              MultilayerPerceptron.sigmoid_derivative(hidden_layer_input))

                # numpy.multiply is scalar, numpy.dot is matrix multiplication, numpy.outer is outer product

                # update weights
                # W = W + alpha * delta_output * hidden_layer_output
                self.output_weights = numpy.add(self.output_weights, numpy.multiply(alpha, numpy.outer(hidden_layer_output, delta_output)))

                # Wh = Wh + alpha * (input_layer*delta_hiddenT)
                self.hidden_weights = numpy.add(self.hidden_weights,
                                                numpy.multiply(alpha, numpy.outer(input_layer, delta_hidden.T)))

                # bo = bo + alpha * delta_output
                self._biases[1] = numpy.add(self._biases[1], numpy.multiply(alpha, delta_output))
                # bn = bn + alpha * delta_hidden
                self._biases[0] = numpy.add(self._biases[0], numpy.multiply(alpha, delta_hidden))

    def test(self, data: numpy.ndarray) -> float:
        """
        Test the network on the given data

        :param data: Test data
        :return: Accuracy rate
        """
        correct = 0
        for row in data:
            input_layer = row[1:]

            # hidden layer output
            hidden_layer_input = numpy.dot(input_layer, self.hidden_weights) + self._biases[0]
            hidden_layer_output = self.sigmoid(hidden_layer_input)

            # generate the full output
            output_layer_input = numpy.dot(hidden_layer_output, self.output_weights) + self._biases[1]
            output_layer_output = self.sigmoid(output_layer_input)

            if self.output_nodes == 1:
                if abs(output_layer_output[0] - row[0]) < 0.5:
                    correct += 1
            else:
                guess_index = numpy.argmax(output_layer_output)
                actual = row[0]
                if guess_index == actual:
                    correct += 1

        return correct / len(data)
