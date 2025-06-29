import pway_funcs as fn2
from sklearn import preprocessing
import scipy.io as scio
import pandas as pd
from mrmr import mrmr_classif
from scipy import stats

import GRAB
from aal_region_name import *

# 获得somfcn
def get_somfcn(filename): # filename = ROISignals_sub.mat所在位置
    data_mat = scio.loadmat(filename) # 读取mat文件将内容加载到字典中
    data = data_mat['ROISignals'][:, :90] # 从字典中提取‘ROIsignals’键对应的值（脑区间的信号强度）
                                            # 值为二维数组，取前90列，因为我们只关注特定的90个脑区
    data, _ = sparse_SMO(data)
    return data

def sparse_SMO(train):
    max_iter = 50 # 最大迭代次数
    tol = 1e-4 # 容忍度，当前解的目标函数值与最优解的目标函数值之间的差距小于容忍度时，停止迭代
    dual_max_iter = 600 # 对偶问题的最大迭代次数
    dual_tol = 1e-5 # 对偶问题的容忍度
                        # 对偶问题通常更易解决，因此在求解原问题时，通常会先求解对偶问题，然后再根据对偶问题的解求出原问题的解
                        # 对偶问题的最优解的值是原问题最优解的下界
    train1 = fn2.standardize(train) # 标准化数据
    data_1 = train1.T # 转置
    S = np.cov(data_1)  # 协方差矩阵
    data = train
    node_num = 90
    (Theta, blocks) = GRAB.BCD_modified(Xtrain=data, Ytrain=data, S=S, lambda_1=16, lambda_2=8, K=5, max_iter=max_iter,
                                        tol=tol, dual_max_iter=dual_max_iter, dual_tol=dual_tol)#块坐标下降方法优化目标函数
    # print("Theta: ", Theta)
    # print("Overlapping Blocks: ", blocks)
    Theta = -Theta
    # print(np.array(Theta).shape)

    Theta = Theta - np.diag(np.diag(Theta)) # 将对角线元素设置为0，而非对角线元素保持不变
    Theta = Theta.reshape(((node_num * node_num), 1)) # 将Theta变为一维数组

    # 数据归一化到[-1,1]，MinAbsScaler为归一化到[0,1]
    max_abs_scaler = preprocessing.MaxAbsScaler() # 将每个特征（在这里是Theta数组的每个元素）除以该特征的最大绝对值，从而实现归一化
    Theta = np.array(Theta) # 将Theta变为numpy数组
    Theta = max_abs_scaler.fit_transform(Theta) # 拟合数据并标准化
    Theta = Theta.reshape((node_num, node_num)) # 将Theta变为原来的二维数组
    return Theta, blocks


def getPerson(x_simple, y_simple):
    # 与dataframe的皮尔逊相关系数操作为是否转置的区别,转置之后结果相等
    return np.corrcoef(x_simple, y_simple) # 皮尔逊相关系数是一种衡量两个变量之间线性相关程度的指标，其值在-1到1之间，1表示完全正相关，-1表示完全负相关，0表示无关。

# 获取高阶FCN 输入为som_fcn的数据
def get_somhfcn(data):
    # (30*116 -> 30*30)
    dd = np.zeros_like(data)
    for i in range(len(data)):
        for j in range(len(data[i])):
            if i < j:
                np.seterr(divide='ignore', invalid='ignore')
                # 计算皮尔逊相关系数
                tmp = getPerson(data[i], data[j])[0][1] # getPerson函数返回一个二维数组，其中[0][1]和[1][0]元素是皮尔逊相关系数
                                                        # [0][0]和[1][1]元素是1，因为一个数组与自身的皮尔逊相关系数是1
                dd[i, j] = 0 if np.isnan(tmp) else tmp  # 如果tmp是nan则返回0，否则返回tmp
            else:
                dd[i, j] = dd[j, i] # 如果i>=j，则dd[i][j] = dd[j][i]，因为皮尔逊相关系数是对称的
    return dd


# 获取混合高阶FCN 输入为som_afcn和som_fcn的数据
def get_associated_FC_control(low_datas, high_datas):
    dd = 0.5 * low_datas + 0.5 * high_datas
    max_dd = np.max(dd)
    dd = abs(dd / max_dd) #归一化到[0,1]
    return dd


# 取上三角，打平后返回
def process_extract_upmatirx_feature_one(Matrix):
    a = np.array([])
    for i in range(Matrix.shape[0] - 1):
        a = np.append(a, Matrix[i + 1, : i + 1])
    a.astype(int)
    return a


# 使用mrmr算法获取数据的特征索引
def getmRMR(data_x, data_y, feature_count=50):
    X = pd.DataFrame(data_x).fillna(0)
    y = pd.Series(data_y)
    selected_features = mrmr_classif(X, y, K=feature_count)  # 索引从0到n-1
    return selected_features


# t检验获得异常连接的索引
# 传入被试的AFCN矩阵(1*4005)，返回异常连接的索引（1-4005）
def get_ttest_idx_AFCN(subject_data):
    # 载入正常人的数据
    normal_data = np.load('data/AFCN_NC.npy')

    #独立样本t检验
    t_values, p_values = stats.ttest_ind(normal_data, subject_data)
    # 排序特征并选择与正常人差异最显著的几个特征
    num_top_features = 5  # 选择差异最显著的前5个特征
    top_indices = np.argsort(p_values)[:num_top_features]

    return top_indices

# t检验获得异常连接的索引
# 传入被试的特征选择后的AFCN矩阵(1*selected_num),特征选择索引，返回异常连接的索引(1-4005)
def get_ttest_idx_AFCN_selected(subject_data, selected_indices, num_top_features):
    # 载入正常人的数据
    normal_data = np.load('data/AFCN_NC.npy')
    normal_data = normal_data[:, selected_indices]
    subject_data = subject_data.reshape(1, -1)
    subject_data = subject_data[:, selected_indices]
    subject_data = subject_data.reshape(-1)

    #独立样本t检验
    t_values, p_values = stats.ttest_ind(normal_data, subject_data)
    # 排序特征并选择与正常人差异最显著的几个特征
    num_top_features = 5  # 选择差异最显著的前5个特征
    top_indices = np.argsort(p_values)[:num_top_features]
    top_indices = selected_indices[top_indices]

    return top_indices

# 被打平的t检验索引转换为矩阵索引
def flatten_index_to_matrix_index(idx):
    n = 90
    length = n - 1
    x = 0
    while idx > length - 1:
        idx -= length
        x += 1
        length -= 1
    y = idx + n - length
    return x + 1, y + 1  # 最后x+1和y+1是因为AAL模板编号从1开始

# 获得异常连接的脑区的中文名称
def get_brain_area_name(i, j):
    ls_LR = get_LR_from_aal()  # 左右脑
    chinese_name = get_Chinese_name()  # 脑区中文名称
    name_i = ls_LR[i] + "." + chinese_name[i]
    name_j = ls_LR[j] + "." + chinese_name[j]
    return name_i, name_j

# 根据索引获得异常连接的脑区的中文名称列表
def get_brain_area_name_list(top_indices):
    # 脑区列表
    brain_area_name_list = []
    for top_index in top_indices:
        i, j = flatten_index_to_matrix_index(top_index)
        name_i, name_j = get_brain_area_name(i, j)
        name_ij = name_i + " - " + name_j
        brain_area_name_list.append(name_ij)
    return brain_area_name_list