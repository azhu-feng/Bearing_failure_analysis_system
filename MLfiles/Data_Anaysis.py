import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


def load_and_preprocess_data(file_path):
    # 加载CSV文件
    data = pd.read_csv(file_path)

    # 提取特征和标签
    X = data.drop('fc', axis=1)  # 特征：最大值、最小值、方差、偏度、峰度、均方根、波形因子
    y = data['fc']  # 标签：故障类型

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集、验证集和测试集
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


def one_hot(Train_Y, Val_Y, Test_Y):
    # 标签转换为二维数组，便于后续编码
    Train_Y = np.array(Train_Y).reshape([-1, 1])
    Val_Y = np.array(Val_Y).reshape([-1, 1])
    Test_Y = np.array(Test_Y).reshape([-1, 1])

    # 创建OneHotEncoder对象 使用Train_Y训练编码器，确定类别范围
    Encoder = preprocessing.OneHotEncoder()
    Encoder.fit(Train_Y)

    # 进行one hot 编码并转换成数组，设置类型为int32
    Train_Y = Encoder.transform(Train_Y).toarray()
    Val_Y = Encoder.transform(Val_Y).toarray()
    Test_Y = Encoder.transform(Test_Y).toarray()

    Train_Y = np.asarray(Train_Y, dtype=np.float32)
    Val_Y = np.asarray(Val_Y, dtype=np.float32)
    Test_Y = np.asarray(Test_Y, dtype=np.float32)
    return Train_Y, Val_Y, Test_Y


def decode_fault_type(encoded_fault_type, fault_type_to_name):
    class_index = np.argmax(encoded_fault_type)  # 对于概率分布，取最大概率对应的索引；对于one-hot编码，直接取值为1的索引
    return fault_type_to_name[class_index]


def load_data(file_path):
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(file_path)

    num_classes = len(np.unique(y_train))  # 确定故障类型的类别数量
    fault_types = list(np.unique(y_train))  # 获取所有唯一的故障类型

    y_train_one_hot, y_val_one_hot, y_test_one_hot = one_hot(y_train, y_val, y_test)
    x_train, x_val, x_test = X_train[:, :, np.newaxis], X_val[:, :, np.newaxis], X_test[:, :, np.newaxis]
    return x_train, x_val, x_test, y_train_one_hot, y_val_one_hot, y_test_one_hot, num_classes, fault_types
