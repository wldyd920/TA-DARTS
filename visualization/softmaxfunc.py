import numpy as np
import math
import pandas as pd
import  torch.nn.functional as F
import matplotlib.pyplot as plt

# logit = np.array([-0.00017602, 0.00066729, -0.00011738, 0.00055787, 0.0020953, 0.0018927, -0.00028905, 0.00026138])
# logit = np.array([0.2812, 0.0596, -0.1521, -0.1574, -0.0032, 0.1826, 0.0467, -0.1012])


# Temperature
# t = 0.1

def softmax_temp(logit, t=1):
    numes = []
    res = []
    for j in logit:
        j_exp = math.exp(j/t)
        numes.append(j_exp)
    deno = sum(numes)
    for i in numes:
        res.append(i/deno)
    return res
    
# res = softmax_temp(logit, t)
# print("분자:",numes)
# print("결과:", res)
# print("기존 알파값:",logit)
# print("이전 합:", sum(logit))
# print("분모:",j_sum)
# print("결과 합 검산:", sum(res))
# print("t =", t)

def softmax_specification(logit, t=1):
    numes = []
    res = []
    for j in logit:
        j_exp = math.exp(j/t)
        numes.append(j_exp)
    deno = sum(numes)
    for i in numes:
        res.append(i/deno)
    return deno, numes, res

def show_softmax(logit, t=1):
    deno, numes, res = softmax_specification(logit, t)
    data = {"alpha": logit, "exp(a/t)": numes,"softmax(a/t)": res}
    df = pd.DataFrame(data)
    print(df)
    print("t = ", t)
    print("sum(alpha):", sum(logit))
    print("sum(exp(a/t)):", deno)
    print("sum(softmax(a/t)):", sum(res))
    return df

# print(show_softmax(logit))

