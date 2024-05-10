from keras.layers import Dense, LSTM, Bidirectional, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import TensorBoard

from MLfiles.Data_Anaysis import load_data, decode_fault_type
from MLfiles.GPU import use_Gpu
from MLfiles.model_analysis import evaluate_model
from MLfiles.loss_acc import loss_acc


def build_cnn_model(num_classes):
    model = Sequential([
        Bidirectional(LSTM(units=64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(units=32)),
        Dropout(0.5),
        Dense(units=num_classes, activation='softmax', kernel_regularizer=l2(1e-4))
    ])

    model.compile(optimizer=Adam(lr=0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# 训练模型
def train_cnn_model(model, X_train, y_train, X_val, y_val, dataclass, epochs=20, batch_size=230):
    tb_cb = TensorBoard(log_dir='C:/Users/z1834/Desktop/Bearing_Failure_Analysis_System/MLfiles/logs/logs-BiLstm')

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                        verbose=1, callbacks=[tb_cb])
    model.save(f'C:/Users/z1834/Desktop/Bearing_Failure_Analysis_System/MLfiles/model/{dataclass}/bilstm.h5')

    return history


def BiLstm(test, dataclass):
    file_path = 'C:/Users/z1834/Desktop/Bearing_Failure_Analysis_System/dataset/uploads/feature.csv'
    model_name = "bilstm"

    x_train, x_val, x_test, y_train, y_val, y_test, num_classes, fault_types = load_data(file_path)

    if test:
        use_Gpu()
        model = build_cnn_model(num_classes)
        history = train_cnn_model(model, x_train, y_train, x_val, y_val, dataclass)
        loss_acc(model_name, history)

    else:
        model = load_model(f'C:/Users/z1834/Desktop/Bearing_Failure_Analysis_System/MLfiles/model/{dataclass}/bilstm.h5')
    acc = evaluate_model(model, x_test, y_test, model_name)

    fault_type_to_name = {i: type_name for i, type_name in enumerate(fault_types)}
    predicted_fault_type_one_hot = model.predict(x_test)[0]  # 模型预测出一个样本的one-hot编码
    predicted_fault_type_name = decode_fault_type(predicted_fault_type_one_hot, fault_type_to_name)
    print("预测的故障类型：", predicted_fault_type_name)
    return predicted_fault_type_name, acc
