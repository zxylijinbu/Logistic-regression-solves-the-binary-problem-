import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1/(1+np.exp(-z))  


def loss(x, y, theta):
    """
    计算损失函数
    :param x: 特征矩阵 5000*1
    :param y: 标签矩阵 1*2
    :param theta:参数
    :return:损失函数
    """
    m, n = x.shape  # M*N
    g = sigmoid(np.dot(x, theta))
    loss_num = (-1.0/m)*np.sum(y.T*np.log(g)+(1-y).T*np.log(1-g))  # 计算损失函数
    return loss_num


def calculate_f1(data_y_, g):
    tp = 0
    fp = 0
    fn = 0

    y_ = np.mat(data_y_)
    m, n = y_.shape
    m_row = 0
    n_col = 0
    while m_row < m:
        n_col = 0
        while n_col < n:
            if g[m_row, n_col:n_col+1] < 0.5:  # 预测结果为负例
                if y_[m_row, n_col:n_col+1] == 1:  # 真实结果为正例
                    fn += 1
            if g[m_row, n_col:n_col+1] >= 0.5:   # 预测结果为正例
                if y_[m_row, n_col:n_col+1] == 1:  # 真实结果为正例
                    tp += 1
                elif y_[m_row, n_col:n_col+1] == 0:  # 真实结果为负例
                    fp += 1
            n_col += 1
        m_row += 1

    precision_ratio = tp/(tp+fp)
    recall_ratio = tp/(tp+fn)
    f1_measure = 2*precision_ratio*recall_ratio/(precision_ratio + recall_ratio)
    print("precision_ratio={} recall_ratio={} f1_measure={}".format(precision_ratio, recall_ratio,f1_measure))
    return precision_ratio, recall_ratio, f1_measure


def calculate_f1_measure(data_x_, data_y_,theta):
    """
    计算f1 measure值
    :param data_x_:特征矩阵 5000*1
    :param data_y_:标签矩阵 1*2
    :param theta:参数
    :return:f1 measure值和loss值
    """
    x_ = np.mat(data_x_)
    y_ = np.mat(data_y_)
    g_new = sigmoid(np.dot(x_, theta))
    precision_ratio, recall_ratio, f1_measure = calculate_f1(y_, g_new)
    loss_ = loss(x_, y_, theta)
    return f1_measure, loss_


def batch_gradient_descent(data_x_train, data_y_train,data_x_validation, data_y_validation, learning_rate, maxepochs=1000, epsilon=1e-6):
    """
    使用批量梯度下降法（Batch gradient descent）求解逻辑回归
    :param data_x_train: 训练集特征矩阵 5000*1
    :param data_y_train:训练集标签矩阵 1*2
    :param data_x_validation: 验证集特征矩阵 5000*1
    :param data_y_validation:验证集标签矩阵 1*2
    :param learning_rate:步长
    :param maxepochs:最大迭代次数
    :param epsilon:损失精度
    :return:参数theta
    """
    x_train = np.mat(data_x_train)  # m*n
    y_train = np.mat(data_y_train)
    m, n = x_train.shape  # m-row,n-col
    theta = np.ones((n, 1))  # N*1 初始化参数theta
    epochs_cnt = 0
    loss_train_list = []
    f1_measure_list_train = []
    f1_measure_list_vali = []
    loss_validation_list = []
    while epochs_cnt < maxepochs:

        loss_train = loss(x_train, y_train, theta)  # 上一次的损失值
        g = sigmoid(np.dot(x_train, theta))  # 预测值
        error = g-y_train  # 预测值与标签值的误差
        grad = (1.0/m) * np.dot(x_train.T, error)  # 损失函数的梯度
        theta = theta-learning_rate*grad

        loss_new_train = loss(x_train, y_train, theta)
        loss_train_list.append(loss_new_train)
        print("epochs:{} loss:{}".format(epochs_cnt, loss_new_train))
        print("train:")
        f1_measure_train, loss_t = calculate_f1_measure(data_x_train, data_y_train, theta)
        f1_measure_list_train.append(f1_measure_train)

        print("validation:")
        f1_measure_vali, loss_validation = calculate_f1_measure(data_x_validation, data_y_validation, theta)
        f1_measure_list_vali.append(f1_measure_vali)
        loss_validation_list.append(loss_validation)

        if abs(loss_new_train-loss_train) < epsilon:
            break
        epochs_cnt += 1

    print('迭代到第{}次，结束迭代！'.format(epochs_cnt))
    plt.figure()
    # plt.gca().set_color_crcle(['blue', 'red'])
    plt.plot(loss_train_list, 'g')
    plt.plot(loss_validation_list, 'r')
    plt.legend(["loss_train", "loss_validation"], loc='upper left')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

    plt.figure()
    plt.plot(f1_measure_list_train, 'g')
    plt.plot(f1_measure_list_vali, 'r')
    plt.legend(["f1_measure_train", "f1_measure_validation"], loc='upper left')
    plt.xlabel('epochs')
    plt.ylabel('f1_measure')
    plt.show()
    return theta


def convert_to_matrix(lines):
    mat_all_1 = []
    mat_all_2 = []
    num = 0
    for line in lines:
        num += 1
        line = line.replace("\n", "")
        line = line.split("  ")

        # 处理5000维向量
        arr1 = line[0]
        arr1 = arr1.replace("[", "")
        arr1 = arr1.replace("]", "")
        arr1 = arr1.replace(",", "")
        arr1_new = []
        for i in arr1:
            if i != " ":
                arr1_new.append(int(i))
        arr1 = np.array(arr1_new)
        if arr1.__len__() == 5000:
            arr1 = arr1.reshape(1, 5000)
            mat_all_1.append(arr1)
        else:
            continue

        # 处理2维向量
        arr2 = line[1]
        arr2 = arr2.replace("[", "")
        arr2 = arr2.replace("]", "")
        arr2 = arr2.replace(",", "")
        arr2_new = []
        for i in arr2:
            if i != " ":
                arr2_new.append(int(i))
        arr2 = np.array(arr2_new)
        if arr2.__len__() == 2:
            arr2 = arr2.reshape(1, 2)
            mat_all_2.append(arr2)
        else:
            continue

    mat_all_1 = np.concatenate(mat_all_1)
    mat_all_2 = np.concatenate(mat_all_2)
    print(num)
    return mat_all_1, mat_all_2


def get_matrix(f_path):
    """
    获取矩阵
    :param f_path: 拥有5000维和2维词向量
    :return: x,y两个矩阵
    """
    file = open(f_path, mode="r", encoding="UTF-8")
    lines = file.readlines()
    mat1, mat2 = convert_to_matrix(lines)
    return mat1, mat2


if __name__ == '__main__':
    path = "C:\\Users\\86185\\Desktop\\bit\\大二下\\知识工程\\作业一\\数据集\\"
    file_path_train = path+"train_raw_ver1.txt"
    file_path_validation = path + "validation_raw_ver1.txt"
    file_path_test = path + "test_raw_ver1.txt"

    datax_train, datay_train = get_matrix(file_path_train)
    datax_validation, datay_validation = get_matrix(file_path_validation)
    datax_test, datay_test = get_matrix(file_path_test)

    theta_bgd = batch_gradient_descent(datax_train, datay_train, datax_validation, datay_validation, learning_rate=10)
    print("test:")
    test_f1_measure, loss_test= calculate_f1_measure(datax_test, datay_test, theta_bgd)
    print("test f1_measure:{}".format(test_f1_measure))


