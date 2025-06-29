import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os
from process_funcs import getmRMR

# 加载data中的数据
AD_data_list = np.load('processed_data/AFCN_AD.npy')
EMCI_data_list = np.load('processed_data/AFCN_EMCI.npy')
LMCI_data_list = np.load('processed_data/AFCN_LMCI.npy')
NC_data_list = np.load('processed_data/AFCN_NC.npy')

AD_data = np.vstack(AD_data_list)
EMCI_data = np.vstack(EMCI_data_list)
LMCI_data = np.vstack(LMCI_data_list)
NC_data = np.vstack(NC_data_list)

# 每类数据的数量
print(AD_data.shape)
print(EMCI_data.shape)
print(LMCI_data.shape)
print(NC_data.shape)

# 生成标签 0-NC 1-EMCI 2-LMCI 3-AD
AD_label = np.ones((AD_data.shape[0], 1)) * 3
EMCI_label = np.ones((EMCI_data.shape[0], 1)) * 1
LMCI_label = np.ones((LMCI_data.shape[0], 1)) * 2
NC_label = np.ones((NC_data.shape[0], 1)) * 0

# 合并四类数据
ALL_data = np.vstack((NC_data, EMCI_data, LMCI_data,AD_data))
# 四类数据对应的标签
ALL_label = np.vstack((NC_label, EMCI_label, LMCI_label, AD_label))
ALL_label = np.ravel(ALL_label) #将多维数组降为一维
print("数据集加载完成")

# mrmr特征选择
selected_feature_indices = getmRMR(ALL_data, ALL_label, 200)
print("mrmr特征选择完成")
print(selected_feature_indices)
# 保存特征索引
np.save('processed_data/selected_feature_200_4.npy', selected_feature_indices)

# 选择特征
selected_data = ALL_data[:, selected_feature_indices]

print('开始训练')
# 特征缩放
scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_data)

# 设置SVM参数网格
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'] + list(np.logspace(-3, 2, 6)),
    'degree': [2, 3, 4],
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

# 使用最佳参数重新训练SVM分类器
best_svm = SVC(**best_params, probability=True)
best_svm.fit(scaled_data, ALL_label)

# 保存最优模型和缩放器到model文件夹
os.makedirs("model", exist_ok=True)
joblib.dump(best_svm, "model/best_svm_4_200_81.2.pkl")
joblib.dump(scaler, "processed_data/scaler_4.pkl")