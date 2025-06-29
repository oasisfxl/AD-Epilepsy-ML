import numpy as np


# 获得 AAL模板中的中文名字
def get_Chinese_name():
    f = open('aal/aal_name_Chinese.txt', encoding="UTF-8")
    names = f.read()
    ls_name = names.split("\n")
    f.close()
    return ls_name


# 获得 AAL模板中的左右脑L R标注列表
def get_LR_from_aal():
    ls = get_simple_name_from_aal()
    ls_LR = []  # 提取左右信息
    for item in ls:
        # ls_LR.append(item.split(".")[1])
        ls_LR.append(item.split(".")[1])
    return ls_LR


# 获得AAL简称
def get_simple_name_from_aal():
    f1 = open('aal/aal_name.txt')
    name1 = f1.read()
    f1.close()
    return name1.split("\n")


# 获得AAL全名称
def get_full_name_from_aal():
    f1 = open('aal/aal_name_full.txt')
    name1 = f1.read()
    f1.close()
    name1 = name1.split("\n")
    return name1

if __name__ == '__main__':

    ls_LR = get_LR_from_aal()

    simple_name = get_simple_name_from_aal()
    chinese_name = get_Chinese_name()
    ls_full_name = get_full_name_from_aal()

    AAL_list = np.array([2, 37, 38, 47, 48, 50, 51, 67, 68, 77, 82, 83])
    for i in AAL_list:
        print(ls_LR[i] + "." + ls_full_name[i])
    print("********************")
    for i in AAL_list:
        print(ls_LR[i] + "." + chinese_name[i])
