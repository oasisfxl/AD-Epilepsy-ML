import time
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score, auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os
from process_funcs import getmRMR
import matplotlib.pyplot as plt


# 加载newdata中的数据
AD_data_list = np.load('processed_data/AFCN_AD.npy')
EMCI_data_list = np.load('processed_data/AFCN_EMCI.npy')
LMCI_data_list = np.load('processed_data/AFCN_LMCI.npy')
NC_data_list = np.load('processed_data/AFCN_NC.npy')

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
# np.save('processed_data/selected_feature_200.npy', selected_feature_indices)

# 选择特征
selected_data = ALL_data[:, selected_feature_indices]

print('开始训练')
# 特征缩放
scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_data)
# #保存scaler对象
# joblib.dump(scaler, 'processed_data/scaler_200.pkl')
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(scaled_data, ALL_label, test_size=0.4, random_state=42)

# 设置SVM参数网格
param_grid = {
    'C': [0.1, 1, 5, 10, 50, 100],
    # 'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto'] + list(np.logspace(-3, 2, 6)),
    # 'degree': [2, 3, 4, 5, 6]
}

# 创建SVM分类器
svm = SVC(kernel='rbf', probability=True)

# 创建十折交叉验证对象
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# 使用GridSearchCV进行参数调整
grid_search = GridSearchCV(svm, param_grid, scoring='accuracy', cv=cv, n_jobs=-1, verbose=1)

# 进行网格搜索
grid_search.fit(X_train, y_train)

# 获取最佳参数和准确率
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("最佳参数：", best_params)
print("最佳准确率：", best_accuracy)

# 使用最佳参数训练SVM分类器并统计训练时间
start_time = time.time()
best_svm = SVC(**best_params, probability=True, kernel='rbf')
best_svm.fit(X_train, y_train)
svm_training_time = time.time() - start_time
print(f"SVM training time: {svm_training_time} seconds")

# 计算正类的概率
y_scores = best_svm.predict(scaled_data)

# # 计算ROC曲线和ROC区域
# fpr, tpr, _ = roc_curve(ALL_label, y_scores, pos_label=3)
# roc_auc = auc(fpr, tpr)

# # 绘制ROC曲线
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()

y_pred = best_svm.predict(X_test)
# 输出classification_report
print(classification_report(y_test, y_pred))

# 使用最佳参数重新训练SVM分类器
best_svm = SVC(**best_params, probability=True)
best_svm.fit(scaled_data, ALL_label)

# # best_svm的路径，best_svm_2_特征数量_最佳准确率
# best_svm_path = "model/best_svm_2_" + str(feature_num) + '_' + str(round(best_accuracy, 3)*100) + ".pkl"

# # 保存最优模型和缩放器到model文件夹
# os.makedirs("./model", exist_ok=True)
# joblib.dump(best_svm, best_svm_path)