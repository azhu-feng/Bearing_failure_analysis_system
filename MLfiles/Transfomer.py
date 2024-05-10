from keras.layers import Dense, BatchNormalization, Dropout, MultiHeadAttention
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from MLfiles.Data_Anaysis import load_data, decode_fault_type
from MLfiles.GPU import use_Gpu
from MLfiles.model_analysis import evaluate_model
from MLfiles.loss_acc import loss_acc

from keras import layers, Model


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads,
                                      value_dim=embed_dim // num_heads)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim), ]
        )
        self.layernorm1 = BatchNormalization()
        self.layernorm2 = BatchNormalization()
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, **kwargs):
        attn_output = self.mha(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=kwargs.get("training"))
        # print("intput", inputs.shape)
        # print("attn:", attn_output.shape)
        out1 = self.layernorm1(inputs + attn_output)
        # print("out1:", out1.shape)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=kwargs.get("training"))
        # print("ffn:", ffn_output.shape)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate
        })
        return config


# 构建Transformer模型
def build_transformer_model(input_shape, num_classes, embed_dim, num_heads, ff_dim, num_layers):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    # print("inputs:", inputs.shape)
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    output = layers.Flatten()(x)
    # 输出层调整为对应类别数的Softmax分类
    outputs = layers.Dense(num_classes, activation="softmax")(output)
    # print("outputs:", outputs.shape)
    return Model(inputs=inputs, outputs=outputs)


def train_trans_model(model, x_train, y_train, x_val, y_val, dataclass, epochs=20, batch_size=230):
    tb_cb = TensorBoard(log_dir='C:/Users/z1834/Desktop/Bearing_Failure_Analysis_System/MLfiles/logs/logs-trans')

    history = model.fit(
        x_train, y_train, epochs=epochs,
        batch_size=230,
        validation_data=(x_val, y_val),
        verbose=1, callbacks=[tb_cb],
        shuffle=True)

    model.save(f'C:/Users/z1834/Desktop/Bearing_Failure_Analysis_System/MLfiles/model/{dataclass}/trans.h5')

    return history


def trans(test, dataclass):
    embed_dim = 256
    num_heads = 20
    ff_dim = 256
    num_layers = 12
    model_name = "trans"
    file_path = 'C:/Users/z1834/Desktop/Bearing_Failure_Analysis_System/dataset/uploads/feature.csv'

    x_train, x_val, x_test, y_train, y_val, y_test, num_classes, fault_types = load_data(file_path)

    if test:
        use_Gpu()
        input_shape = x_train.shape[1:]
        model = build_transformer_model(input_shape, num_classes, embed_dim, num_heads, ff_dim, num_layers)
        model.compile(optimizer=Adam(0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
        history = train_trans_model(model, x_train, y_train, x_val, y_val, dataclass)
        loss_acc(model_name, history)

    # initial_learning_rate = 0.001
    # lr_schedule = ExponentialDecay(
    #     initial_learning_rate,
    #     decay_steps=10000,
    #     decay_rate=0.96)
    # optimizer = Adam(lr=lr_schedule)

    # earlyStop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, mode='max', verbose=1,
    #                           restore_best_weights=True)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    else:
        custom_objects = {'TransformerBlock': TransformerBlock}
        model = load_model(f'C:/Users/z1834/Desktop/Bearing_Failure_Analysis_System/MLfiles/model/{dataclass}/trans.h5', custom_objects=custom_objects)
    acc = evaluate_model(model, x_test, y_test, model_name)

    fault_type_to_name = {i: type_name for i, type_name in enumerate(fault_types)}
    predicted_fault_type_one_hot = model.predict(x_test)[0]  # 模型预测出一个样本的one-hot编码
    predicted_fault_type_name = decode_fault_type(predicted_fault_type_one_hot, fault_type_to_name)
    print("预测的故障类型：", predicted_fault_type_name)
    return predicted_fault_type_name, acc
