# 调用get_somfcn函数获得somfcn(低阶的FCN)
# 调用get_somhfcn函数获得somhfcn(高阶的FCN)
# get_associated_FC_control函数获得somafcn(混合FCN)
# 调用process_extract_upmatirx_feature_one函数得到1*4005的矩阵，保存该矩阵
# 加载特征索引，特征选择
# 加载模型，进行预测，0-NC，1-EMCI，2-LMCI，3-AD
# 病灶分析得到异常连接的脑区中文名称列表
# 返回一个字典，包含预测结果、准确率、异常连接的脑区中文名称列表和错误信息

from process_funcs import *
import joblib

def main_process(filename):
    # 调用get_somfcn函数获得somfcn(低阶的FCN)
    somfcn = get_somfcn(filename)
    # 调用get_somhfcn函数获得somhfcn(高阶的FCN)
    somhfcn = get_somhfcn(somfcn)
    # get_associated_FC_control函数获得somafcn(混合FCN)
    somafcn = get_associated_FC_control(somfcn, somhfcn)
    # 调用process_extract_upmatirx_feature_one函数得到1*4005的矩阵
    upmatrix = process_extract_upmatirx_feature_one(somafcn)
    upmatrix = upmatrix.reshape(1, -1)
    # upmatrix保存(之后保存到数据库中)
    # 加载特征索引，特征选择(old)
    feature_index = np.load('data/selected_feature_50.npy')
    upmatrix_selected = upmatrix[:, feature_index]
    # 加载模型，进行预测
    model = joblib.load('model/best_svm_2_50_97.0.pkl')
    result = model.predict(upmatrix_selected)

    # 预测结果，0-NC，1-EMCI，2-LMCI，3-AD
    result_dict = {'0': 'NC', '1': 'EMCI', '2': 'LMCI', '3': 'AD'}
    result = result_dict[str(int(result[0]))]

    # 返回一个字典，包含预测结果、准确率、异常连接的脑区中文名称列表和错误信息
    if result != 'NC':
        # 病灶分析得到异常连接的脑区中文名称列表
        top_indices = get_ttest_idx_AFCN_selected(upmatrix, feature_index, num_top_features=5)
        brain_area_name_list = get_brain_area_name_list(top_indices)
        return {'result': result, 'brain_area_name_list': brain_area_name_list, 'error': '处理成功',
                'accuracy': '98.7%'}
    else:
        return {'result': result, 'brain_area_name_list': '无异常连接', 'error': '处理成功', 'accuracy': '98.7%'}


if __name__ == '__main__':
    filename = 'data/ROISignals_168_S_6754.mat'  #Roisignals_sub.mat文件路径
    response = main_process(filename)
    print(response)