import numpy as np

def sigmoid(x, derivative=False):
    if derivative:
        sig = sigmoid(x)
        return sig * (1 - sig)
    return 1 / (1 + np.exp(-x))

def tanh(x, derivative=False):
    if derivative:
        return 1 - np.tanh(x)**2
    return np.tanh(x)

def relu(x, derivative=False):
    if derivative:
        return np.where(x > 0, 1, 0)
    return np.maximum(0, x)

def softmax(x, derivative=False):
    if derivative:
        s = softmax(x).reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)
