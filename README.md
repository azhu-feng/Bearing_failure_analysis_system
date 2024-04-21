tensorboard --logdir="C:/Users/z1834/Desktop/Bearing_Failure_Analysis_System/MLfiles/logs/logs-Cnn"
tensorboard --logdir="C:/Users/z1834/Desktop/Bearing_Failure_Analysis_System/MLfiles/logs/logs-Lstm"
tensorboard --logdir="C:/Users/z1834/Desktop/Bearing_Failure_Analysis_System/MLfiles/logs/logs-BiLstm"

电机轴承故障预警系统的研究与实现，先对三个数据集（美国西储大学数据集，江南大学数据集，美国-机械故障预防技术学会MFPT数据集）进行数据分析，
得出最大值、最小值、平均值、方差、均方根、偏度、峰度、峰值因子、波形因子、脉冲因子。
再用热力图，折线图，散点图观察分析这些特征，
最后保留最大值、最小值、平均值、方差、均方根、偏度这些数据，并将其保存在CSV文件里。
再用django写一个前端页面，从这个页面传入CSV文件，后端接收到CSV文件后再用CNN,LSTM,BiLSTM进行故障预测，
结果发现对于其中两个数据集准确率LSTM＞BiLSTM＞CNN，其中一个是LSTM=BiLSTM＞CNN。

 if test:
        use_Gpu()
        model = build_cnn_model(input_shape, num_classes)
        history = train_cnn_model(model, x_train, y_train, x_val, y_val, dataclass)
        loss_acc(model_name, history)

    else:
        if dataclass == 0:
            model = load_model("C:/Users/z1834/Desktop/Bearing_Failure_Analysis_System/MLfiles/model/CRWU/cnn.h5")
        elif dataclass == 1:
            model = load_model("C:/Users/z1834/Desktop/Bearing_Failure_Analysis_System/MLfiles/model/JNU/cnn.h5")
        else:
            model = load_model("C:/Users/z1834/Desktop/Bearing_Failure_Analysis_System/MLfiles/model/MFPT/cnn.h5")
        acc = evaluate_model(model, x_test, y_test, model_name)

   if dataclass == 0:
        model.save('C:/Users/z1834/Desktop/Bearing_Failure_Analysis_System/MLfiles/model/CRWU/cnn.h5')
    elif dataclass == 1:
        model.save('C:/Users/z1834/Desktop/Bearing_Failure_Analysis_System/MLfiles/model/JNU/cnn.h5')
    else:
        model.save('C:/Users/z1834/Desktop/Bearing_Failure_Analysis_System/MLfiles/model/MFPT/cnn.h5')