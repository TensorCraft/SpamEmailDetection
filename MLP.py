import numpy as np
from Activations import sigmoid, relu, softmax
from lossfunction import binary_cross_entropy, categorical_cross_entropy, mse_loss
import tensorflow as tf # to be removed, test only
from tensorflow.keras.datasets import mnist # to be removed, test only

class MLP:
    def __init__(self, layers, lr=0.001, lossfunction=categorical_cross_entropy):
        self.layers = layers
        self.weights = []
        self.biases = []
        self.lr = lr
        self.lossfn = lossfunction

        for i in range(len(layers) - 1):
            input_size = layers[i][0]
            output_size = layers[i + 1][0]
            self.weights.append(np.random.randn(input_size, output_size) * 0.01)
            self.biases.append(np.zeros((1, output_size)))

    def forward(self, x):
        activations = x
        activations_list = [x]  # 保存每层的激活值，用于反向传播
        for i in range(len(self.layers) - 1):
            z = np.dot(activations, self.weights[i]) + self.biases[i]
            activations = self.layers[i + 1][1](z)  # 计算下一层的激活值
            activations_list.append(activations)
        return activations_list

    def learn(self, x, y):
        # 前向传播
        activations_list = self.forward(x)
        activations = activations_list[-1]

        # 计算损失和梯度
        loss = self.lossfn(y, activations)
        gradient_loss = self.lossfn(y, activations, gradient=True)
        print("Loss:", loss)
        
        # 反向传播
        delta = gradient_loss
        for i in range(len(self.layers) - 2, -1, -1):
            dW = np.dot(activations_list[i].T, delta)
            db = np.sum(delta, axis=0, keepdims=True)
            
            self.weights[i] -= self.lr * dW
            self.biases[i] -= self.lr * db

            if i != 0:
                delta = np.dot(delta, self.weights[i].T) * self.layers[i][1](activations_list[i], derivative=True)

        return activations

if __name__ == "__main__":
    # 加载MNIST数据集
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0  # 将数据reshape为(sample_size, 28*28)并归一化
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

    # 处理y_train和y_test
    y_train = np.eye(10)[y_train]  # one-hot encoding
    y_test = np.eye(10)[y_test]

    mlp = MLP(layers=[(28*28, relu), (1024, relu), (512, sigmoid), (10, softmax)], lossfunction=categorical_cross_entropy)

    # 使用批处理进行训练
    batch_size = 64
    for epoch in range(10):  # 设置训练轮数
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            mlp.learn(x_batch, y_batch)
