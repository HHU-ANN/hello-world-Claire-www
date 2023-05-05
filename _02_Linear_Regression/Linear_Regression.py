# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os
try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np
def ridge(data, alpha):
    X = np.column_stack((np.ones(len(data[:, :-1])), data[:, :-1]))
    y = data[:, -1]
    coef = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + alpha * np.identity(X.shape[1])), X.T), y)
    return coef
def lasso(data, alpha, lr=0.001, max_iter=10000):
    X = np.column_stack((np.ones(len(data[:, :-1])), data[:, :-1]))
    y = data[:, -1]
    w = np.zeros(X.shape[1])
    for i in range(max_iter):
        grad = np.dot(X.T, np.dot(X, w) - y) + alpha * np.sign(w)
        w -= lr * grad
    return w
def read_data(path='./data/exp02/'):
    X_train = np.load(path + 'X_train.npy')
    y_train = np.load(path + 'y_train.npy')
    return np.column_stack((X_train, y_train))
data = read_data()
# 岭回归求解
coef_ridge = ridge(data, alpha=0.1)
print(coef_ridge)
# Lasso回归求解
coef_lasso = lasso(data, alpha=0.1)
print(coef_lasso)
