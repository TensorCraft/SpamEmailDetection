import numpy as np

def binary_cross_entropy(y_true, y_pred, epsilon=1e-15, gradient=False):
    """
    计算二元交叉熵损失

    参数:
    y_true -- 实际标签，形状为 (n_samples,)
    y_pred -- 预测概率，形状为 (n_samples,)
    epsilon -- 防止log(0)的极小值

    返回值:
    loss -- 二元交叉熵损失
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # 防止log(0)
    if gradient:
        return (y_pred - y_true) / (y_pred * (1 - y_pred))
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def mse_loss(y_true, y_pred, gradient=False):
    """
    计算均方误差损失

    参数:
    y_true -- 真实目标值
    y_pred -- 模型预测值

    返回值:
    loss -- 均方误差损失
    """
    if gradient:
        return 2 * (y_pred - y_true) / y_true.size
    return np.mean(np.power(y_true - y_pred, 2))

def categorical_cross_entropy(y_true, y_pred, epsilon=1e-15, gradient=False):
    """
    计算分类交叉熵损失

    参数:
    y_true -- 真实目标值（one-hot编码）
    y_pred -- 预测概率
    epsilon -- 防止log(0)的极小值

    返回值:
    loss -- 分类交叉熵损失
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    if gradient:
        return y_pred - y_true
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return loss
