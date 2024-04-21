from keras.utils import plot_model


# 评估模型
def evaluate_model(model, X_test, y_test, model_name):
    scores = model.evaluate(X_test, y_test, verbose=0)
    plot_model(model=model, to_file=f"C:/Users/z1834/Desktop/Bearing_Failure_Analysis_System/MLfiles/model_picture/{model_name}.png", show_shapes=True)
    print("Test accuracy: {:.2f}%".format(scores[1] * 100))
    acc = "{:.2f}".format(scores[1] * 100)
    return acc


# 预测故障类型
def predict_fault_type(model, X_test):
    y_pred = model.predict_classes(X_test)
    return y_pred
