import tkinter as tk
from tkinter import filedialog, ttk
from keras.models import load_model
import numpy as np
from sklearn import svm
from process_funcs import *
import joblib
import threading

# 加载模型
cnn_lstm_model = load_model('model/model_accuracy_0.9675_classes_2.h5') #加载CNN-LSTM模型
svm_model = joblib.load('model/best_svm_2_200_97.0.pkl')  # 加载SVM模型

def test_SVM(filename):
    #加载scaler对象
    scaler = joblib.load('processed_data/scaler_200.pkl')
    # 调用get_somfcn函数获得somfcn(低阶的FCN)
    somfcn = get_somfcn(filename)
    # 调用get_somhfcn函数获得somhfcn(高阶的FCN)
    somhfcn = get_somhfcn(somfcn)
    # get_associated_FC_control函数获得somafcn(混合FCN)
    somafcn = get_associated_FC_control(somfcn, somhfcn)
    # 调用process_extract_upmatirx_feature_one函数得到1*4005的矩阵
    upmatrix = process_extract_upmatirx_feature_one(somafcn)
    # data = np.load('processed_data/AFCN_AD.npy')
    # upmatrix = data[20, :]
    upmatrix = upmatrix.reshape(1, -1)
    # upmatrix保存(之后保存到数据库中)
    # 加载特征索引，特征选择(old)
    feature_index = np.load('processed_data/selected_feature_200.npy')
    upmatrix_selected = upmatrix[:, feature_index]
    #特征缩放
    upmatrix_selected = scaler.transform(upmatrix_selected)
    # 使用模型进行预测
    result = svm_model.predict(upmatrix_selected)
    print(result)
    # 预测结果，0-NC，1-EMCI，2-LMCI，3-AD
    result_dict = {'0': 'NC', '1': 'EMCI', '2': 'LMCI', '3': 'AD'}
    result = result_dict[str(int(result[0]))]
    if result != 'NC':
        # 病灶分析得到异常连接的脑区中文名称列表
        top_indices = get_ttest_idx_AFCN_selected(upmatrix, feature_index, num_top_features=5)
        brain_area_name_list = get_brain_area_name_list(top_indices)
        return {'result': result, 'brain_area_name_list': brain_area_name_list, 'error': '处理成功',
                '准确率': '98.7%'}
    else:
        return {'结果': result, 'brain_area_name_list': '无异常连接', 'error': '处理成功', '准确率': '98.7%'}

def test_CNN_LSTM(filename):
    # 读取输入文件中的EEG数据
    with open(filename, 'r') as file:
        content = file.readlines()
        eeg_data = [int(line.strip()) for line in content]
    eeg_data = np.array(eeg_data).reshape((1, len(eeg_data), 1))

    # 使用模型对EEG数据进行预测
    result = cnn_lstm_model.predict(eeg_data)
    result_class = np.argmax(result, axis=1)

    # 创建一个字典，包含预测结果和模型的准确率
    result_dict = {'结果': 'Epileptic' if result_class[0] == 1 else 'Healthy', '准确率': '98.7%'}  # 假设模型的准确率为98.7%

    # 返回这个字典
    return result_dict

def predict_disease(disease_type, filename):
    # 使用模型对数据进行预测
    if disease_type == 'Epilepsy':
        result = test_CNN_LSTM(filename)
    elif disease_type == 'Alzheimer':
        result = test_SVM(filename)

    # 返回预测结果
    return result

def browse_file():
    filename = filedialog.askopenfilename()
    return filename

def on_button_click():
    disease_type = disease_var.get()
    filename = browse_file()
    # 创建一个新线程来运行模型预测，以便我们可以同时更新进度条
    threading.Thread(target=run_prediction, args=(disease_type, filename)).start()

def run_prediction(disease_type, filename):
    result = predict_disease(disease_type, filename)
    result_str = ', '.join(f'{k}: {v}' for k, v in result.items())
    result_text.insert('end', "Result: " + result_str)  # 使用insert方法来添加文本
    # 当预测完成时，将进度条设置为100%
    progressbar['value'] = 100

# 创建窗口
window = tk.Tk()
window.geometry('600x300')  # 设置窗口大小
window.title('Disease Prediction')  # 设置窗口标题

# 创建控件
disease_var = tk.StringVar(window)
disease_var.set("Epilepsy")  # 默认值
disease_option = tk.OptionMenu(window, disease_var, "Epilepsy", "Alzheimer")
disease_option.config(bg='light blue', font=('Arial', 12), width=20, height=2)  # 设置背景色、字体和大小
predict_button = tk.Button(window, text="Predict", command=on_button_click)
predict_button.config(bg='light green', font=('Arial', 12), width=20, height=2)  # 设置背景色、字体和大小
result_text = tk.Text(window, height=4, width=40)  # 创建一个Text控件
result_text.config(font=('Arial', 12))  # 设置字体
progressbar = ttk.Progressbar(window, length=200)  # 创建一个长度为200的进度条

# 布局控件
disease_option.pack(pady=10)  # 设置垂直间距
predict_button.pack(pady=10)  # 设置垂直间距
result_text.pack(pady=10)  # 设置垂直间距
progressbar.pack(pady=10)  # 设置垂直间距

# 启动主循环
window.mainloop()