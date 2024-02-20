import pandas as pd
import numpy as np

from softmaxfunc import softmax_temp

# raw data load
path = "C:/Users/user/VSC/DARTS/dragen0/alpha/"

def alpha_list(path):
    f = open(path+"alphas.txt", 'r')
    lines = f.readlines()

    # data reformat
    edge_start = 0
    epoch = []
    for line in lines:
        new_line = line.replace('[', '')
        new_line = new_line.replace(']', '')
        new_line = new_line.replace(' ', '')
        new_line = new_line.replace('\n', '')
        new_line = new_line.split(',')
        # print(len(new_line))
        
        # # 알파값 저장하고 싶으면 여기서 가능 (norm reduc 구분 안되있음) 
        # # transform to string first if needed
        # with open(path+"alphas2.txt", 'w') as fi:
        #     fi.write(new_line)

        # edge와 arch, epoch로 3차원 리스트
        edge = []
        arch = []
        for number in new_line:
            new_number = float(number)
            # print(new_number)
            if edge_start < 7 :
                edge.append(new_number)
                edge_start += 1
            elif edge_start == 7:
                edge.append(new_number)
                arch.append(edge)
                edge_start = 0
                edge = []
        epoch.append(arch)

    # # 확인
    # print(epoch[2][0])
    # print(len(epoch))

    # alpha_normal 과 alpha_reduce로 나눔
    alpha_normals = []
    alpha_reduces = []
    divider = 0
    for i in epoch:
        if divider == 0:
            alpha_normals.append(i)
            divider = 1
        elif divider == 1:
            alpha_reduces.append(i)
            divider = 0

    # # 확인
    # print(alpha_normals[1][0])
    # print(alpha_reduces[0][0])

    return alpha_normals, alpha_reduces


def alpha_extraction(path, what_alpha, epoch, edge):
    # 각 edge별로 확인 (변수 지정 필요)
    alpha_normals, alpha_reduces = alpha_list(path)
    # what_alpha = alpha_reduces

    # print("edge:", what_alpha[which_epoch][which_edge])
    # print("edge의 alpha 합:", sum(what_alpha[which_epoch][which_edge][:]))

    return what_alpha[epoch][edge]


def alpha_df(path):
    alpha_normals, alpha_reduces = alpha_list(path)
    cnt = 0
    normal_dic = {}
    for epoch in alpha_normals:
        normal_dic[f"{cnt}"] = epoch
        cnt+=1
    normal_df = pd.DataFrame(normal_dic)

    cnt = 0
    reduce_dic = {}
    for epoch in alpha_reduces:
        reduce_dic[f"{cnt}"] = epoch
        cnt+=1
    reduce_df = pd.DataFrame(reduce_dic)

    return normal_df, reduce_df

# # 확인
# nor, red = alpha_df(path)
# print(red['10'][1])


def beta_df(path, t=1):
    alpha_normals, alpha_reduces = alpha_list(path)
    cnt = 0
    normal_dic = {}
    for epoch in alpha_normals:
        beta_epoch = []
        for edge in epoch:
            beta_edge = softmax_temp(edge, t)
            beta_epoch.append(beta_edge)
        normal_dic[f"{cnt}"] = beta_epoch
        cnt+=1
    beta_normal_df = pd.DataFrame(normal_dic)

    cnt = 0
    reduce_dic = {}
    for epoch in alpha_reduces:
        beta_epoch = []
        for edge in epoch:
            beta_edge = softmax_temp(edge, t)
            beta_epoch.append(beta_edge)
        reduce_dic[f"{cnt}"] = beta_epoch
        cnt+=1
    beta_reduce_df = pd.DataFrame(reduce_dic)

    return beta_normal_df, beta_reduce_df


# # to excel
beta_nor, beta_red = beta_df(path, 1)
print(beta_red["10"][1])
# beta_nor.to_excel('beta_normals.xlsx')
# beta_red.to_excel('beta_reduces.xlsx')
