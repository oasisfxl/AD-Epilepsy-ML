import time
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os
from process_funcs import getmRMR
from sklearn.linear_model import LogisticRegression

# 加载newdata中的数据
AD_data_list = np.load('data/AFCN_AD.npy')
EMCI_data_list = np.load('data/AFCN_EMCI.npy')
LMCI_data_list = np.load('data/AFCN_LMCI.npy')
NC_data_list = np.load('data/AFCN_NC.npy')

AD_data = np.vstack(AD_data_list)
EMCI_data = np.vstack(EMCI_data_list)
LMCI_data = np.vstack(LMCI_data_list)
NC_data = np.vstack(NC_data_list)

# print(AD_data.shape)
# print(EMCI_data.shape)
# print(LMCI_data.shape)
# print(NC_data.shape)

# 生成标签 0-NC 1-EMCI 2-LMCI 3-AD
AD_label = np.ones((AD_data.shape[0], 1)) * 3
EMCI_label = np.ones((EMCI_data.shape[0], 1)) * 1
LMCI_label = np.ones((LMCI_data.shape[0], 1)) * 2
NC_label = np.ones((NC_data.shape[0], 1)) * 0

# 合并两类数据
ALL_data = np.vstack((NC_data, AD_data))
# 两类数据对应的标签
ALL_label = np.vstack((NC_label, AD_label))
ALL_label = np.ravel(ALL_label) #将多维数组降为一维
print("数据集加载完成")

# mrmr特征选择
feature_num = 200
selected_feature_indices = getmRMR(ALL_data, ALL_label, feature_num)
print("mrmr特征选择完成")
print(selected_feature_indices)
# # 保存特征索引
# np.save('data/selected_feature_30.npy', selected_feature_indices)

# 选择特征
selected_data = ALL_data[:, selected_feature_indices]

print('开始训练')
# 特征缩放
scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_data)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(scaled_data, ALL_label, test_size=0.2, random_state=42)

# 创建逻辑回归分类器
lr = LogisticRegression()

# 设置逻辑回归参数网格
param_grid = {
    'C': [0.1, 1, 10, 100, 1000], # 惩罚参数，控制了误分类的惩罚，值越大，对误分类的惩罚越大，越容易过拟合
    'penalty': ['l1', 'l2'], # 惩罚项，l1为L1正则化，l2为L2正则化
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], # 优化算法
}

# 创建十折交叉验证对象
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# 使用GridSearchCV进行参数调整
grid_search = GridSearchCV(lr, param_grid, scoring='accuracy', cv=cv, n_jobs=-1, verbose=1)

# 进行网格搜索
grid_search.fit(scaled_data, ALL_label)

# 获取最佳参数和准确率
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("最佳参数：", best_params)
print("最佳准确率：", best_accuracy)

# 使用最佳参数训练逻辑回归分类器并统计训练时间
start_time = time.time()
best_lr = LogisticRegression(**best_params)
best_lr.fit(X_train, y_train)
lr_training_time = time.time() - start_time
print(f"LR training time: {lr_training_time} seconds")

y_pred = best_lr.predict(X_test)
# 输出classification_report
print(classification_report(y_test, y_pred))

#使用最佳参数重新训练逻辑回归分类器
best_lr = LogisticRegression(**best_params)
best_lr.fit(scaled_data, ALL_label)

# best_lr的路径，best_lr_2_特征数量_最佳准确率
best_lr_path = "model/best_lr_2_" + str(feature_num) + '_' + str(round(best_accuracy, 3)*100) + ".pkl"

# 保存最优模型到model文件夹
joblib.dump(best_lr, best_lr_path)