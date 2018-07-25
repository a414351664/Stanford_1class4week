# encoding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from dnn_app_utils_v2 import *

# ５层的神经网络随机初始化时×１效果99％，　76％过拟合　　　　×２效果65％，　34％
# ７层的神经网络随机初始化时×１效果99％，　74％过拟合　　　　×２效果65％，　34％
def two_layer_model(x, y, layer_dims, learning_rate, iter_num):
    np.random.seed(1)
    n_x = layer_dims[0]
    n_h = layer_dims[1]
    n_y = layer_dims[2]
    para = initialize_parameters(n_x, n_h, n_y)
    costs = []
    # Z, cache = linear_forward(x)
    for i in range(iter_num):
        W1 = para["W1"]
        b1 = para["b1"]
        W2 = para["W2"]
        b2 = para["b2"]
        # forward compute
        A = x
        # 只有三层的神经网
        A_pre = A
        A, cache1 = linear_activation_forward(A_pre, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A, W2, b2, "sigmoid")

        # cost_compute
        cost = compute_cost_reg(A2, y, W1, W2)
        # backforward compute
        dA2 = -np.divide(y, A2) + (np.divide(1-y, 1-A2))
        dA1, dW2, db2 = linear_activation_backward_reg(dA2, cache2, 'sigmoid')
        dA0, dW1, db1 = linear_activation_backward_reg(dA1, cache1, 'relu')
        grads = {"dW1":dW1, "dW2":dW2, "db1":db1, "db2":db2}
        para = update_parameters(para, grads, learning_rate)
        if i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            costs.append(cost)
    return para, costs

def L_layer_model(x, y, layer_dims, learning_rate, iter_num):
    np.random.seed(1)
    para = initialize_parameters_deep(layer_dims)
    # para = initialize_parameters_deep(layer_dims)
    costs = []
    # Z, cache = linear_forward(x)
    for i in range(iter_num):
        # forward compute
        # 多层的神经网
        AL, caches = L_model_forward(x, para)
        # cost_compute
        cost = compute_cost(AL, y)
        # backforward compute
        grads = L_model_backward(AL, y, caches)
        para = update_parameters(para, grads, learning_rate)
        if i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            costs.append(cost)
    return para, costs

def main():
    # 返回的是一个字典，描述图片的属性
    plt.rcParams["figure.figsize"] = (5.0, 4.0)
    plt.rcParams["image.interpolation"] = 'nearest'
    plt.rcParams["image.cmap"] = 'gray'
    np.random.seed(1)
    # 显示一张猫的图片
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_data()
    # y:(1, m) x:(m, nx)
    # index = 7
    # plt.imshow(train_set_x_orig[index])
    # print("y = ", train_set_y_orig[0, index], "It's a ", classes[train_set_y_orig[0, index]].decode("utf-8"))
    # plt.show()

    # 得到训练的样本，图片的分辨率(209, 64, 64, 3)    (1, 209)  test只有50个数据
    # m_train = train_set_x_orig.shape[0]
    # num_px = train_set_x_orig.shape[1]
    # m_test = test_set_x_orig.shape[1]
    train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    train_x = train_set_x / 255.
    test_x = test_set_x / 255.
    n_x = train_x.shape[0]
    n_h = 7
    n_y = len(train_set_y_orig)

    # layer_dims = (n_x, n_h, n_y)
    # learning_rate = 0.0075
    # para, costs= two_layer_model(train_x, train_set_y_orig, layer_dims, learning_rate, iter_num=1500)
    # # analysize the lost
    # plt.plot(np.squeeze(costs))
    # plt.xlabel("iteration per tens")
    # plt.ylabel("cost")
    # plt.title("Learning rate is " + str(learning_rate))
    # plt.show()
    # pre = predict(train_x, train_set_y_orig, para)
    # pre_test = predict(test_x, test_set_y_orig, para)
    # 以上为三层的神经网络，下面设置五层的

    layer_dims = (n_x, 20, 7, 5, n_y)
    learning_rate = 0.01
    para, costs = L_layer_model(train_x, train_set_y_orig, layer_dims, learning_rate, iter_num=2000)
    plt.plot(np.squeeze(costs))
    plt.xlabel("iteration per tens")
    plt.ylabel("cost")
    plt.title("Learning rate is " + str(learning_rate))
    plt.show()
    pre = predict(train_x, train_set_y_orig, para)
    pre_test = predict(test_x, test_set_y_orig, para)
    print_mislabeled_images(classes, test_x, test_set_y_orig, pre_test)
    pass

if __name__ == '__main__':
    main()