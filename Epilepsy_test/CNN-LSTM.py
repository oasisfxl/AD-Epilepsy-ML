import os

from keras.src.utils import to_categorical
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dropout, BatchNormalization, LSTM
import matplotlib.pyplot as plt
from keras import regularizers
from keras.models import load_model
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.metrics import confusion_matrix

import pandas as pd
from sklearn.model_selection import train_test_split
from zipfile import ZipFile
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam

extract_path = 'BoonData'  # 提取路径

# 读取和预处理数据 用于读取指定目录下的数据文件，并将其内容和对应的标签一起保存。
def read_data(directory_path, label):
    data = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r') as file:
            content = file.readlines()
            content = [int(line.strip()) for line in content]
        data.append((content, label))
    return data

all_data = []
# labels = ['F', 'N', 'O', 'S', 'Z']  # 标签列表
labels = ['Healthy', 'Epileptic']  # 标签列表
healthy_labels = ['Z', 'O']
epileptic_labels = ['F', 'N', 'S']

# for label in labels:
#    directory_path = os.path.join(extract_path, label)
#    all_data.extend(read_data(directory_path, label))

for label in healthy_labels:
    directory_path = os.path.join(extract_path, label)
    all_data.extend(read_data(directory_path, 'Healthy'))

for label in epileptic_labels:
    directory_path = os.path.join(extract_path, label)
    all_data.extend(read_data(directory_path, 'Epileptic'))

df = pd.DataFrame(all_data, columns=['features', 'label'])
features = np.array(df['features'].tolist())
labels = pd.Categorical(df['label']).codes

# 划分数据集
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, random_state=42)

# 重塑数据集
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
X_val_reshaped = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
# print(X_train_reshaped.shape)

Y_train = to_categorical(Y_train, num_classes=2)
Y_test = to_categorical(Y_test, num_classes=2)
Y_val = to_categorical(Y_val, num_classes=2)

# 创建一个Sequential模型
model = Sequential()

# 添加3个卷积块
conv_blocks = [40, 80, 150]
for i in range(3):
    model.add(Conv1D(conv_blocks[i], 7, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(Dropout(0.2)) #丢弃概率为20%
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))

# 添加3层LSTM层
model.add(LSTM(300, return_sequences=True))
model.add(LSTM(300, return_sequences=True))
model.add(LSTM(300))

# 添加3个全连接块
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(2, activation='softmax'))  # 假设分类任务的类别总数为2
# 编译模型 交叉熵作为损失函数，Adam作为优化器
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999), metrics=['accuracy'])

# 使用训练集进行训练
history = model.fit(X_train_reshaped, Y_train, validation_data=(X_val_reshaped, Y_val), epochs=300, batch_size=128) # 300次迭代，批次大小128

# # 获取模型在训练集上的最后一个epoch的准确率
# accuracy = history.history['accuracy'][-1]
#
# # 获取模型的分类数量
# num_classes = Y_train.shape[1]
#
# # 使用准确率和分类数量来命名模型
# model_name = f'model_accuracy_{accuracy:.4f}_classes_{num_classes}.h5'
#
# # 保存模型
# model.save(model_name)

# 使用模型预测测试集的标签
Y_pred = model.predict(X_test_reshaped)

# 将预测结果转换为类别标签
Y_pred_classes = np.argmax(Y_pred, axis=1)

# 将测试集的标签也转换为类别标签
Y_test_classes = np.argmax(Y_test, axis=1)

# 使用sklearn库的classification_report来评估模型的性能
print(classification_report(Y_test_classes, Y_pred_classes))

# 使用sklearn库的confusion_matrix来查看模型的混淆矩阵
# 第i行第j列的元素表示实际属于第i类但被预测为第j类的样本数量。
print(confusion_matrix(Y_test_classes, Y_pred_classes))

# 使用模型预测测试集的概率
Y_pred_prob = model.predict(X_test_reshaped)

# 计算ROC曲线
fpr, tpr, _ = roc_curve(Y_test_classes, Y_pred_prob[:, 1])

# 计算AUC
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()