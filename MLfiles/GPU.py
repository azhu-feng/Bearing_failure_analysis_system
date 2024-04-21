import tensorflow as tf

# print(tf.test.is_gpu_available())
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def use_Gpu():
    # 列出可用的 GPU 设备
    print(tf.config.list_physical_devices('GPU'))

    # 设置 TensorFlow 使用 GPU 进行计算
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)
