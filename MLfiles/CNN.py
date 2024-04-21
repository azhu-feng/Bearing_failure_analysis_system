from keras.layers import Dense, Conv1D, BatchNormalization, MaxPooling1D, Activation, Flatten, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import TensorBoard

from MLfiles.Data_Anaysis import load_data, decode_fault_type
from MLfiles.GPU import use_Gpu
from MLfiles.model_analysis import evaluate_model
from MLfiles.loss_acc import loss_acc


def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        # 第一层卷积
        Conv1D(filters=16, kernel_size=64, strides=16, padding='same', kernel_regularizer=l2(1e-4),
               input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2, padding='same'),

        # 第二层卷积
        Conv1D(32, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2, padding='same'),

        # 卷积到全连续需要展平
        Flatten(),
        Dropout(0.2),

        # 添加全连接层
        Dense(32),
        Activation("relu"),

        # 增加输出层，共num_classes个单元
        Dense(units=num_classes, activation='softmax', kernel_regularizer=l2(1e-4))
    ])

    model.compile(optimizer=Adam(lr=0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# 训练模型
def train_cnn_model(model, X_train, y_train, X_val, y_val, dataclass, epochs=8, batch_size=230):
    tb_cb = TensorBoard(log_dir='C:/Users/z1834/Desktop/Bearing_Failure_Analysis_System/MLfiles/logs/logs-Cnn')

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                        verbose=1, callbacks=[tb_cb], shuffle=True)

    model.save(f'C:/Users/z1834/Desktop/Bearing_Failure_Analysis_System/MLfiles/model/{dataclass}/cnn.h5')

    return history


def CNN(test, dataclass):
    file_path = 'C:/Users/z1834/Desktop/Bearing_Failure_Analysis_System/dataset/uploads/feature.csv'
    model_name = "cnn_1D"

    x_train, x_val, x_test, y_train, y_val, y_test, num_classes, fault_types = load_data(file_path)

    if test:
        use_Gpu()
        input_shape = x_train.shape[1:]  # 原始统计特征数据的形状为 (num_features, 1)
        model = build_cnn_model(input_shape, num_classes)
        history = train_cnn_model(model, x_train, y_train, x_val, y_val, dataclass)
        loss_acc(model_name, history)

    else:
        model = load_model(f'C:/Users/z1834/Desktop/Bearing_Failure_Analysis_System/MLfiles/model/{dataclass}/cnn.h5')
    acc = evaluate_model(model, x_test, y_test, model_name)

    fault_type_to_name = {i: type_name for i, type_name in enumerate(fault_types)}
    predicted_fault_type_one_hot = model.predict(x_test)[0]  # 模型预测出一个样本的one-hot编码
    predicted_fault_type_name = decode_fault_type(predicted_fault_type_one_hot, fault_type_to_name)
    print("预测的故障类型：", predicted_fault_type_name)
    return predicted_fault_type_name, acc

