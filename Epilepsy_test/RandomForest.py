import os
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump

extract_path = 'BoonData'  # 提取路径

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
healthy_labels = ['Z', 'O']
epileptic_labels = ['F', 'N', 'S']

for label in healthy_labels:
    directory_path = os.path.join(extract_path, label)
    all_data.extend(read_data(directory_path, 'Healthy'))

for label in epileptic_labels:
    directory_path = os.path.join(extract_path, label)
    all_data.extend(read_data(directory_path, 'Epileptic'))

df = pd.DataFrame(all_data, columns=['features', 'label'])
features = np.array(df['features'].tolist())
labels = pd.Categorical(df['label']).codes

X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, Y_train)

Y_pred = rf.predict(X_test)

# 计算模型的准确率
accuracy = accuracy_score(Y_test, Y_pred)

# 获取分类的数量
num_classes = len(np.unique(Y_test))

# 使用准确率和分类数量来命名模型
model_name = f'randomforest_model_accuracy_{accuracy:.4f}_classes_{num_classes}.joblib'

# 保存模型
dump(rf, model_name)

print(classification_report(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))