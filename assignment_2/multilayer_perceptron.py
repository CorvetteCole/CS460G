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
                        numpy.random.uniform(low=-1, high=1, size=1)]

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

    def train(self, data: numpy.ndarray, alpha: float) -> "MultilayerPerceptron":
        """
        Train the network using backpropagation

        :param data: Training data
        :param alpha: Learning rate
        :return: self
        """

        # worth noting that the data is a numpy array of shape (n, 785) where n is the number of training examples

        with logging_redirect_tqdm():
            for row in tqdm(data):
                input_layer = row[1:]

                # Generate network output (forward pass)
                hidden_layer_input = numpy.dot(input_layer, self.hidden_weights)
                hidden_layer_input = numpy.add(hidden_layer_input, self._biases[0])
                hidden_layer_output = self.sigmoid(hidden_layer_input)

                # generate the full output
                output_layer_input = numpy.dot(hidden_layer_output, self.output_weights)
                output_layer_output = self.sigmoid(output_layer_input)

                # Calculate output error
                output_error = row[0] - output_layer_output

                # Compute delta at the output
                output_delta = output_error * output_layer_output * (1 - output_layer_output)

                # Propagate delta to previous layer
                hidden_error = numpy.dot(output_delta, self.output_weights.T)

                # Compute delta at the hidden layer
                hidden_delta = hidden_error * hidden_layer_output * (1 - hidden_layer_output)

                # Update weights between previous layer and current layer
                self.output_weights += alpha * numpy.dot(hidden_layer_output.reshape(self.hidden_nodes, 1),
                                                         output_delta.reshape(1, 1))

                # Update weights between input and first hidden layer
                self.hidden_weights += alpha * numpy.dot(input_layer.reshape(self.num_features, 1),
                                                         hidden_delta.reshape(1, self.hidden_nodes))

                # Update biases
                self._biases[0] += alpha * hidden_delta

                self._biases[1] += alpha * output_delta

        return self

    def test(self, data: numpy.ndarray) -> float:
        """
        Test the network on the given data

        :param data: Test data
        :return: Accuracy rate
        """

        # IMPORTANT: we are checking the accuracy of the network on the test data. We will store how many times the network
        # correctly predicts the label of the test data in the variable correct. We will then divide correct by the number
        # of test examples to get the accuracy rate.

        correct = 0
        for row in data:
            input_layer = row[1:]

            # Generate network output (forward pass)
            hidden_layer_input = numpy.dot(input_layer, self.hidden_weights)
            hidden_layer_input = numpy.add(hidden_layer_input, self._biases[0])
            hidden_layer_output = self.sigmoid(hidden_layer_input)

            # generate the full output
            output_layer_input = numpy.dot(hidden_layer_output, self.output_weights)
            output_layer_output = self.sigmoid(output_layer_input)

            # Calculate output error
            output_error = row[0] - output_layer_output

            if output_layer_output > 0.5:
                output_layer_output = 1
            else:
                output_layer_output = 0

            if output_layer_output == row[0]:
                correct += 1

        return correct / len(data)
