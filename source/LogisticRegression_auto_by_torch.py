import torch
import numpy as np
import matplotlib.pyplot as plt


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, H):
        super(LogisticRegressionModel, self).__init__()
        # 定义模型结构
        self.linear = torch.nn.Linear(5000, 2)  # 输入5000维，输出2维

    def forward(self, x):
        g = torch.nn.functional.sigmoid(self.linear(x))
        return g


def f1_scores(probs, label, H):
    probs = probs.detach().numpy()
    label = label.detach().numpy()
    tp = 0
    fp = 0
    fn = 0
    m_row = 0
    while m_row < H:
        n_col = 0
        while n_col < 2:
            if probs[m_row][n_col] < 0.5:  # 预测结果为负例
                if label[m_row][n_col] == 1:  # 真实结果为正例
                    fn += 1
            if probs[m_row][n_col] >= 0.5:   # 预测结果为正例
                if label[m_row][n_col] == 1:  # 真实结果为正例
                    tp += 1
                elif label[m_row][n_col] == 0:  # 真实结果为负例
                    fp += 1
            n_col += 1
        m_row += 1
    precision_ratio = 0
    recall_ratio = 0
    f1_measure = 0
    if tp != 0:
        precision_ratio = tp / (tp + fp)
        recall_ratio = tp / (tp + fn)
        f1_measure = 2 * precision_ratio * recall_ratio / (precision_ratio + recall_ratio)
        print("precision_ratio={} recall_ratio={} f1_measure={}".format(precision_ratio, recall_ratio, f1_measure))
    else:
        print("division by zero!")
    return precision_ratio, recall_ratio, f1_measure


def iterator(x_train, y_train, x_validation, y_validation, x_test, y_test, H_train, H_vali, H_test, maxepoch):
    model = LogisticRegressionModel(H_train)
    epoch_cnt = 0
    loss_fn = torch.nn.BCELoss(size_average=False)  # 计算目标值和预测值之间的二进制交叉熵损失函数
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降,lr为学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam 算法 ,对比梯度下降算法更高效的优化算法

    loss_train_list = []
    loss_vali_list = []
    f1_measure_list_train = []
    f1_measure_list_vali = []

    while epoch_cnt < maxepoch:
        epoch_cnt += 1

        g_train = model(x_train)  # 前向传播
        loss_num_train = loss_fn(g_train, y_train)  # 损失函数 train
        loss_train_list.append(loss_num_train.item())
        print("epochs:{} loss:{}".format(epoch_cnt, loss_num_train.item()))
        print("train:")
        precision_ratio_train, recall_ratio_train, f1_measure_train = f1_scores(g_train, y_train, H_train)
        f1_measure_list_train.append(f1_measure_train)

        optimizer.zero_grad()  # 求导之前把 gradient 清空
        loss_num_train.backward()  # 后向传播 backward pass
        optimizer.step()  # 参数更新

        # 验证集
        g_vali = model(x_validation)
        loss_num_vali = loss_fn(g_vali, y_validation)
        loss_vali_list.append(loss_num_vali.item())
        print("validation:")
        precision_ratio_vali, recall_ratio_vali, f1_measure_vali = f1_scores(g_vali, y_validation, H_vali)
        f1_measure_list_vali.append(f1_measure_vali)

    # 测试集计算f1_measure
    g_test = model(x_test)
    print("test:")
    precision_ratio_test, recall_ratio_test, f1_measure_test = f1_scores(g_test, y_test, H_test)

    # 绘制loss函数图，包含train和validation
    plt.figure()
    plt.plot(loss_train_list, 'g')
    plt.plot(loss_vali_list, 'r')
    plt.legend(["loss_train", "loss_validation"], loc='upper left')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

    # 绘制f1 measure函数图，包含train和validation
    plt.figure()
    plt.plot(f1_measure_list_train, 'g')
    plt.plot(f1_measure_list_vali, 'r')
    plt.legend(["f1_measure_train", "f1_measure_validation"], loc='upper left')
    plt.xlabel('epochs')
    plt.ylabel('f1_measure')
    plt.show()


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
            # len1 = arr1.__len__()
            arr1 = arr1.reshape(1, 5000)
            # print(arr1, "arr1", type(arr1))
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
            # len2 = arr2.__len__()
            arr2 = arr2.reshape(1, 2)
            mat_all_2.append(arr2)
        else:
            continue

    mat_all_1 = np.concatenate(mat_all_1)
    mat_all_2 = np.concatenate(mat_all_2)
    print(num)
    return mat_all_1, mat_all_2, num


def get_matrix(f_path):
    """
    获取矩阵
    :param f_path: 拥有5000维和2维词向量
    :return: x,y两个矩阵
    """
    file = open(f_path, mode="r", encoding="UTF-8")
    lines = file.readlines()
    mat1, mat2, num = convert_to_matrix(lines)
    return mat1, mat2, num


if __name__ == '__main__':
    path = "C:\\Users\\86185\\Desktop\\bit\\大二下\\知识工程\\作业一\\数据集\\"
    file_path_train = path + "train_raw_ver1.txt"
    file_path_validation = path + "validation_raw_ver1.txt"
    file_path_test = path + "test_raw_ver1.txt"

    datax_train, datay_train, H_train = get_matrix(file_path_train)  # H表示有多少组数据
    datax_validation, datay_validation, H_vali = get_matrix(file_path_validation)
    datax_test, datay_test, H_test = get_matrix(file_path_test)

    datax_train = torch.tensor(datax_train, dtype=torch.float32)
    datay_train = torch.tensor(datay_train, dtype=torch.float32)
    datax_validation = torch.tensor(datax_validation, dtype=torch.float32)
    datay_validation = torch.tensor(datay_validation, dtype=torch.float32)
    datax_test = torch.tensor(datax_test, dtype=torch.float32)
    datay_test = torch.tensor(datay_test, dtype=torch.float32)

    iterator(datax_train, datay_train, datax_validation, datay_validation, datax_test, datay_test, H_train, H_vali, H_test, maxepoch=1000)


