import numpy as np


def _entropy(prob):
    prob = prob[prob > 0]
    return np.sum(-prob * np.log2(prob))


if __name__ == "__main__":
    out = _entropy(np.array([0.2, 0.5, 0.3, 0]))
    print(out)
