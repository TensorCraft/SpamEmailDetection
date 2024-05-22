import numpy as np

import numpy as np

def relu(x, derivative=False):
    if derivative:
        return np.where(x > 0, 1, 0)
    return np.maximum(0, x)

def sigmoid(x, derivative=False):
    if derivative:
        sig = sigmoid(x)
        return sig * (1 - sig)
    return 1 / (1 + np.exp(-x))


def softmax(x, derivative=False):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    softmax_vals = e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    if derivative:
        # 计算softmax的导数
        # 使用激活值计算梯度时，可以简化为：(y_pred - y_true)
        return softmax_vals * (1 - softmax_vals)
    
    return softmax_vals