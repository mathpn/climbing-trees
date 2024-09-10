from collections import Counter

import numpy as np


def _entropy(prob):
    prob = prob[prob > 0]
    return np.sum(-prob * np.log2(prob))


def _class_probabilities(labels):
    total_count = len(labels)
    return np.array(
        [label_count / total_count for label_count in Counter(labels).values()]
    )


def _find_best_split(feature, y):
    sort_idx = np.argsort(feature)
    feature_sort = feature[sort_idx]
    y_sort = y[sort_idx]

    min_split_entropy = 1e9
    best_split = 0
    for idx in range(1, len(sort_idx) - 1):
        entropy_l = _entropy(_class_probabilities(y_sort[:idx]))
        entropy_r = _entropy(_class_probabilities(y_sort[idx:]))
        p_l = (idx) / len(sort_idx)
        p_r = (len(sort_idx) - idx) / len(sort_idx)
        conditional_entropy = p_l * entropy_l + p_r * entropy_r
        if conditional_entropy < min_split_entropy:
            min_split_entropy = conditional_entropy
            best_split = feature_sort[idx]

    return best_split


if __name__ == "__main__":
    X_0 = np.random.normal(0, size=100)
    X_1 = np.random.normal(2, size=100)
    X = np.concatenate((X_0, X_1))
    y = np.repeat([0, 1], 100)
    out = _find_best_split(X, y)
    pred = (X >= out).astype(int)
    print(pred)
    print(y)
