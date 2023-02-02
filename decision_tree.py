import numpy as np


def _generate_thresholds(feature):
    return np.percentile(feature, np.arange(0, 100, 25))


def _split(feature, threshold):
    left_index = np.where(feature <= threshold)[0]
    right_index = np.where(feature > threshold)[0]
    return left_index, right_index


def _entropy(y_values):
    p = np.bincount(y_values) / len(y_values)
    entropy = -np.sum([pp * np.log2(pp) for pp in p if pp > 0])
    return entropy


def _information_gain(y, left_index, right_index):
    p = len(y) / (len(left_index) + len(right_index))
    entropy_before = _entropy(y)
    entropy_after = p * _entropy(y[left_index]) + (1 - p) * _entropy(y[right_index])
    return entropy_before - entropy_after


def _best_split(x_values, y_values):
    best_gain = -1
    best_feature_index = None
    best_threshold = None

    for feature_index in range(x_values.shape[1]):
        for threshold in _generate_thresholds(x_values[:, feature_index]):
            gain = _information_gain(y_values, *_split(x_values[:, feature_index], threshold))
            if gain > best_gain:
                best_gain = gain
                best_feature_index = feature_index
                best_threshold = threshold

    return best_feature_index, best_threshold


class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, x_values, y_values):
        self.tree = self._build_tree(x_values, y_values, depth=0)

    def _build_tree(self, x_values, y_values, depth):
        n_samples, n_features = x_values.shape
        n_classes = len(np.unique(y_values))

        # Check if the tree has reached its maximum depth
        if depth >= self.max_depth or n_classes == 1:
            leaf = {
                'prediction': np.bincount(y_values).argmax()
            }
            return leaf

        feature_index, threshold = _best_split(x_values, y_values)
        left_index, right_index = _split(x_values[:, feature_index], threshold)

        left = self._build_tree(x_values[left_index], y_values[left_index], depth + 1)
        right = self._build_tree(x_values[right_index], y_values[right_index], depth + 1)

        node = {
            'feature_index': feature_index,
            'threshold': threshold,
            'left': left,
            'right': right
        }

        return node

    def _predict_sample(self, x):
        node = self.tree
        while 'prediction' not in node:
            if x[node['feature_index']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['prediction']

    def predict(self, x_values):
        y_pred = [self._predict_sample(x) for x in x_values]
        return y_pred
