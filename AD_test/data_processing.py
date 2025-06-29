import glob
import scipy.io as scio
import numpy as np
import re
import process_funcs
# 脑网络稀疏化 阈值选择
def sparse_Graph_threshold(data):  # 传入矩阵
    threshold = 0.1
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] < threshold or data[i, j] == 1:
                data[i, j] = 0
    # print(data)
    return data


# 获取矩阵的上三角 并且打平成1维向量
def get_upmatric(data):  # 传入矩阵
    a = np.array([])
    for i in range(data.shape[0] - 1):
        a = np.append(a, data[i + 1, :i + 1])
    a.astype(int)
    return a


# 保存打平后的上三角矩阵，全部特征
def all_features_extraction(data, root, data_file_name):  # 保存路径，矩阵，文件名
    data = get_upmatric(data)

    features_all = np.array(data)
    savepath = root + '/' + data_file_name + '.npy'
    # 存在feature_classes中
    np.save(savepath, features_all)
    return features_all


# 返回类似002_S_1280的字符串
def get_subnameFromfilename(filename):
    subname = re.findall(r"ROISignals_([0-9].+?).mat", filename)
    return str(subname[0])


# 处理ROICorrelation的数据并保存得到全部特征量
# 总目录，数据标签类型，保存的路径
def process_ROICorrelation_data(origin_root, data_class, processed_root):
    dir_data = origin_root + '/' + data_class + '/ROISignals_FunImgARCWSF'
    all_data_mat_path = glob.glob(dir_data + "/ROICorrelation_[0-9]*.mat")
    for filename in all_data_mat_path:
        data_mat = scio.loadmat(filename)
        data_npy = data_mat['ROICorrelation'][:90, :90]
        data_npy = sparse_Graph_threshold(data_npy)
        subname = get_subnameFromfilename(filename=filename)
        save_dir = processed_root + '/' + data_class
        all_features_extraction(data_npy, save_dir, subname)
        print(data_class + ':' + subname + '----success')


# 处理ROIsignal的数据得到每一类的AFCN矩阵
# 总目录，数据标签类型，保存的路径
def process_ROIsignal_data_L(origin_root, data_class, processed_root):
    new_data_name = 'LFCN_' + data_class + '.npy'
    dir_data = origin_root + '/' + data_class + '/ROISignals_FunImgARCWSF'
    all_data_mat_path = glob.glob(dir_data + "/ROISignals_[0-9]*.mat")
    new_data = []
    for filename in all_data_mat_path:
        data_mat = scio.loadmat(filename)
        data_npy = data_mat['ROISignals'][:, :90]
        data_npy, _ = process_funcs.sparse_SMO(data_npy)# somfcn
        # data_h_npy = process_funcs.get_somhfcn(data_npy)# somhfcn
        # data_npy = process_funcs.get_associated_FC_control(data_npy, data_h_npy)# AFCN
        data_npy = process_funcs.process_extract_upmatirx_feature_one(data_npy)# 打平为一维
        # 将data_npy加入到new_data里
        new_data.append(data_npy)
        subname = get_subnameFromfilename(filename=filename)
        print(data_class + ':' + subname + '----success')
    savepath = processed_root + '/' + new_data_name
    print(new_data)
    np.save(savepath,new_data)

def process_ROIsignal_data_H(origin_root, data_class, processed_root):
    new_data_name = 'HFCN_' + data_class + '.npy'
    dir_data = origin_root + '/' + data_class + '/ROISignals_FunImgARCWSF'
    all_data_mat_path = glob.glob(dir_data + "/ROISignals_[0-9]*.mat")
    new_data = []
    for filename in all_data_mat_path:
        data_mat = scio.loadmat(filename)
        data_npy = data_mat['ROISignals'][:, :90]
        data_npy, _ = process_funcs.sparse_SMO(data_npy)# somfcn
        data_h_npy = process_funcs.get_somhfcn(data_npy)# somhfcn
        # data_npy = process_funcs.get_associated_FC_control(data_npy, data_h_npy)# AFCN
        data_npy = process_funcs.process_extract_upmatirx_feature_one(data_h_npy)# 打平为一维
        # 将data_npy加入到new_data里
        new_data.append(data_npy)
        subname = get_subnameFromfilename(filename=filename)
        print(data_class + ':' + subname + '----success')
    savepath = processed_root + '/' + new_data_name
    print(new_data)
    np.save(savepath,new_data)

def process_ROIsignal_data(origin_root, data_class, processed_root):
    new_data_name = 'AFCN_' + data_class + '.npy'
    dir_data = origin_root + '/' + data_class + '/ROISignals_FunImgARCWSF'
    all_data_mat_path = glob.glob(dir_data + "/ROISignals_[0-9]*.mat")
    new_data = []
    for filename in all_data_mat_path:
        data_mat = scio.loadmat(filename)
        data_npy = data_mat['ROISignals'][:, :90]
        data_npy, _ = process_funcs.sparse_SMO(data_npy)# somfcn
        data_h_npy = process_funcs.get_somhfcn(data_npy)# somhfcn
        data_npy = process_funcs.get_associated_FC_control(data_npy, data_h_npy)# AFCN
        data_npy = process_funcs.process_extract_upmatirx_feature_one(data_npy)# 打平为一维
        # 将data_npy加入到new_data里
        new_data.append(data_npy)
        subname = get_subnameFromfilename(filename=filename)
        print(data_class + ':' + subname + '----success')
    savepath = processed_root + '/' + new_data_name
    print(new_data)
    np.save(savepath,new_data)

origin_root = r'MRI_fMRI_results'
processed_root = r'processed_data'
data_classes = ['AD', 'EMCI', 'LMCI', 'NC']
for data_class in data_classes:
    process_ROIsignal_data_L(origin_root, data_class, processed_root)
    process_ROIsignal_data_H(origin_root, data_class, processed_root)
    print(data_class + 'is finshed')
