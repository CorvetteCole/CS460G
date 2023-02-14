import numpy
import utils


class TreeNode:

    def __init__(self, children=None, feature_index=None, threshold=None):
        if children is None:
            self.children = []

        self.feature_index = feature_index
        self.threshold = threshold
        self.prediction = None

    @property
    def leaf(self) -> bool:
        return len(self.children) == 0


class DecisionTree:
    """
    ## Example data ##
    x_value [10.58,-0.055609]
    y_value True

    The x_value could be a list of any length
    """

    def __init__(self, features: numpy.ndarray, labels: numpy.ndarray, max_depth=3):
        self.max_depth = max_depth
        self.root = self.fit(features, labels)

    def fit(self, features: numpy.ndarray, labels: numpy.ndarray, depth=0) -> TreeNode:
        """
        Fit the decision tree to the data

        :param features: A numpy array of shape (n_samples, n_features)
        :param labels: A numpy array of shape n_samples that are either True or False
        :param depth: The current depth of the tree
        :return: A TreeNode
        """

        # Check if we have reached the maximum depth or if the entropy is 0
        if depth == self.max_depth or utils.calculate_entropy(labels) == 0:
            # set prediction
            node = TreeNode()
            node.prediction = utils.get_majority_label(labels)
            return node

        # Find the best split
        best_feature_index, best_threshold = utils.find_best_split(features, labels)

        # Split the data
        features_left, labels_left, features_right, labels_right = utils.split_data(features, labels,
                                                                                    best_feature_index, best_threshold)

        # Create the node
        node = TreeNode(feature_index=best_feature_index, threshold=best_threshold)

        # Create the left child
        left_child = self.fit(features_left, labels_left, depth + 1)
        node.children.append(left_child)

        # Create the right child
        right_child = self.fit(features_right, labels_right, depth + 1)
        node.children.append(right_child)

        return node

    def predict_label(self, feature: numpy.ndarray) -> bool:
        """
        Predict the label for the given feature

        :param feature: A numpy array of shape (n_features)
        :return: A boolean that represents the predicted label
        """
        node = self.root

        while not node.leaf:
            if feature[node.feature_index] <= node.threshold:
                node = node.children[0]
            else:
                node = node.children[1]

        return node.prediction
