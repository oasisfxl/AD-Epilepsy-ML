import time

import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
import joblib
import os
from sklearn.multiclass import OneVsRestClassifier
from process_funcs import getmRMR
import matplotlib.pyplot as plt

# 加载data中的数据
AD_data_list = np.load('data/AFCN_AD.npy')
EMCI_data_list = np.load('data/AFCN_EMCI.npy')
LMCI_data_list = np.load('data/AFCN_LMCI.npy')
NC_data_list = np.load('data/AFCN_NC.npy')

AD_data = np.vstack(AD_data_list)
EMCI_data = np.vstack(EMCI_data_list)
LMCI_data = np.vstack(LMCI_data_list)
NC_data = np.vstack(NC_data_list)

# 生成标签 0-NC 1-MCI 2-AD
AD_label = np.ones((AD_data.shape[0], 1)) * 2
EMCI_label = np.ones((EMCI_data.shape[0], 1)) * 1
LMCI_label = np.ones((LMCI_data.shape[0], 1)) * 1
NC_label = np.ones((NC_data.shape[0], 1)) * 0

# # 合并四类数据
# ALL_data = np.vstack((NC_data, EMCI_data, LMCI_data,AD_data))
# # 四类数据对应的标签
# ALL_label = np.vstack((NC_label, EMCI_label, LMCI_label, AD_label))
# ALL_label = np.ravel(ALL_label) #将多维数组降为一维

# 合并MCI数据
MCI_data = np.vstack((EMCI_data, LMCI_data))
MCI_label = np.vstack((EMCI_label, LMCI_label))


# 合并三类数据
ALL_data = np.vstack((NC_data, MCI_data, AD_data))
# 三类数据对应的标签
ALL_label = np.vstack((NC_label, MCI_label, AD_label))
ALL_label = np.ravel(ALL_label) #将多维数组降为一维
print("数据集加载完成")

# mrmr特征选择
feature_num = 200
selected_feature_indices = getmRMR(ALL_data, ALL_label, feature_num)
print("mrmr特征选择完成")
print(selected_feature_indices)
# 保存特征索引
np.save('processed_data/selected_feature_200_3.npy', selected_feature_indices)

# 选择特征
selected_data = ALL_data[:, selected_feature_indices]

print('开始训练')
# 特征缩放
scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_data)
#保存scaler对象
joblib.dump(scaler, 'processed_data/scaler_200_3.pkl')

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(scaled_data, ALL_label, test_size=0.2, random_state=42)

param_grid = {
    'C': [0.1, 1, 5, 10, 50, 100],#惩罚参数，控制了误分类的惩罚，值越大，对误分类的惩罚越大，越容易过拟合
    'kernel': ['linear', 'rbf', 'poly'],#核函数，rbf为高斯核函数，linear为线性核函数，poly为多项式核函数
    'gamma': ['scale', 'auto'] + list(np.logspace(-3, 2, 6)),#定义了单个训练样本的影响范围，值越大，影响范围越小，越容易过拟合
    'degree': [2, 3, 4, 5],#定义了多项式核函数的阶数
}

# 创建SVM分类器
svm = SVC(probability=True)

# 创建十折交叉验证对象
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# 使用GridSearchCV进行参数调整
grid_search = GridSearchCV(svm, param_grid, scoring='accuracy', cv=cv, n_jobs=-1, verbose=1)

# 进行网格搜索
grid_search.fit(scaled_data, ALL_label)

# 获取最佳参数和准确率
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("最佳参数：", best_params)
print("最佳准确率：", best_accuracy)

# 使用最佳参数训练SVM分类器并统计训练时间
start_time = time.time()
best_svm = SVC(**best_params, probability=True)
best_svm.fit(X_train, y_train)
svm_training_time = time.time() - start_time
print(f"SVM_3 training time: {svm_training_time} seconds")

# 对测试集进行预测
y_pred = best_svm.predict(X_test)

# 计算并输出总体准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Test Accuracy: {accuracy}")
# 输出精确率、召回率、F1值
report = classification_report(y_test, y_pred)
print(report)

#使用最佳参数重新训练SVM分类器
best_svm = SVC(**best_params, probability=True)
best_svm.fit(scaled_data, ALL_label)
#
# # best_svm的路径，best_svm_3_特征数量_最佳准确率
# best_svm_path = "model/best_svm_3_" + str(feature_num) + '_' + str(round(best_accuracy, 3)*100) + ".pkl"
#
# # 保存最优模型到model文件夹
# os.makedirs("./model", exist_ok=True)
# joblib.dump(best_svm, best_svm_path)