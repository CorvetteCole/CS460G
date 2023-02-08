import numpy as np


class TreeNode:

    def __init__(self, children=None):
        if children is None:
            self.children = []

        self.feature_index = feature_index
        self.threshold = threshold

    def is_leaf(self):
        return len(self.children) == 0


def _calculate_entropy(y_values: np.ndarray) -> float:
    """
    Calculate the entropy for the given labels

    :param y_values: A numpy array of shape n_samples that are either True or False
    :return: The entropy
    """
    # calculate rate of occurrence of True or False present in y_values
    probability = np.bincount(y_values) / len(y_values)
    # calculate entropy
    entropy = -np.sum([p * np.log2(p) for p in probability if p > 0])
    return entropy





class DecisionTree:
    """
    ## Example data ##
    x_value [10.58,-0.055609]
    y_value True

    The x_value could be a list of any length
    """

    def __init__(self, x_values: np.ndarray, y_values: np.ndarray, max_depth=3):
        self.max_depth = max_depth
        self.root = self.fit(x_values, y_values)

    def fit(self, x_values: np.ndarray, y_values: np.ndarray, depth=0) -> TreeNode:
        """
        Fit the decision tree to the data

        :param x_values: A numpy array of shape (n_samples, n_features)
        :param y_values: A numpy array of shape n_samples that are either True or False
        :param depth: The current depth of the tree
        :return: A TreeNode
        """
        if depth == self.max_depth:
            return TreeNode()

        pass

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the given data

        :param x_values: A numpy array of shape (n_samples, n_features)
        :return: A numpy array of shape n_samples that are either True or False
        """
        pass
