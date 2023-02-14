import numpy


def calculate_entropy(labels: numpy.ndarray) -> float:
    """
    Measures the amount of uncertainty in a probability distribution

    :param labels: A numpy array of shape n_samples that are either True or False
    :return: The entropy of the labels
    """
    # calculate rate of occurrence of True or False present in y_values
    probability = numpy.bincount(labels) / len(labels)
    # calculate entropy
    entropy = -numpy.sum([p * numpy.log2(p) for p in probability if p > 0])
    return entropy


def calculate_information_gain(features: numpy.ndarray, labels: numpy.ndarray, feature_index: int, threshold: float):
    """
    Calculates the information gain of a split on a feature

    :param features: A numpy array of shape (n_samples, n_features)
    :param labels: A numpy array of shape n_samples that are either True or False
    :param feature_index: The index of the feature to split on
    :param threshold: The threshold to split on
    :return: The information gain of the split
    """
    # calculate entropy of the whole dataset
    entropy = calculate_entropy(labels)

    # calculate entropy of the left and right split
    left_entropy = calculate_entropy(labels[features[:, feature_index] <= threshold])
    right_entropy = calculate_entropy(labels[features[:, feature_index] > threshold])

    # calculate information gain
    information_gain = entropy - (len(labels[features[:, feature_index] <= threshold]) / len(labels)) * left_entropy - (
            len(labels[features[:, feature_index] > threshold]) / len(labels)) * right_entropy
    return information_gain


def find_best_split(features: numpy.ndarray, labels: numpy.ndarray) -> (int, float):
    """
    Finds the best split for the given data

    :param features: A numpy array of shape (n_samples, n_features)
    :param labels: A numpy array of shape n_samples that are either True or False
    :return: A tuple of the best feature index and threshold
    """
    best_feature_index = 0
    best_threshold = 0
    best_information_gain = 0

    # Find the best split
    for feature_index in range(features.shape[1]):
        # Find the unique values of the feature
        unique_values = numpy.unique(features[:, feature_index])

        # Find the best threshold for the feature
        for threshold in unique_values:
            # Calculate the information gain
            information_gain = calculate_information_gain(features, labels, feature_index, threshold)

            # Check if the information gain is better than the best information gain
            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_feature_index = feature_index
                best_threshold = threshold

    return best_feature_index, best_threshold


def split_data(features: numpy.ndarray, labels: numpy.ndarray, best_feature_index, best_threshold):
    """
    Splits the data based on the best feature and threshold

    :param features: A numpy array of shape (n_samples, n_features)
    :param labels: A numpy array of shape n_samples that are either True or False
    :param best_feature_index: The index of the best feature
    :param best_threshold: The threshold of the best feature
    :return: A tuple of the left features, left labels, right features, and right labels
    """
    features_left = features[features[:, best_feature_index] <= best_threshold]
    labels_left = labels[features[:, best_feature_index] <= best_threshold]

    features_right = features[features[:, best_feature_index] > best_threshold]
    labels_right = labels[features[:, best_feature_index] > best_threshold]

    return features_left, labels_left, features_right, labels_right


def get_majority_label(labels: numpy.ndarray) -> bool:
    """
    Returns the majority label

    :param labels: A numpy array of shape n_samples that are either True or False
    :return: The majority label
    """
    return numpy.bincount(labels).argmax()
