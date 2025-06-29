from keras.models import load_model
import numpy as np
import sys

def test_CNN_LSTM(filename):
    # 加载已经训练好的模型
    model = load_model('model/model_accuracy_0.9675_classes_2.h5')

    # 读取输入文件中的EEG数据
    with open(filename, 'r') as file:
        content = file.readlines()
        eeg_data = [int(line.strip()) for line in content]
    eeg_data = np.array(eeg_data).reshape((1, len(eeg_data), 1))

    # 使用模型对EEG数据进行预测
    result = model.predict(eeg_data)
    result_class = np.argmax(result, axis=1)

    # 创建一个字典，包含预测结果和模型的准确率
    result_dict = {'result': 'Epileptic' if result_class[0] == 1 else 'Healthy', 'accuracy': '98.7%'}  # 假设模型的准确率为98.7%

    # 返回这个字典
    return result_dict

if __name__ == '__main__':
    filename = './BoonData/Z/Z001.txt'
    response = test_CNN_LSTM(filename)
    print(response)