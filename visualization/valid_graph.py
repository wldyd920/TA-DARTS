val_acc = [51.020000,
        64.544000,
        68.768000,
        73.376000,
        74.724000,
        75.472000,
        78.092000,
        78.120000,
        78.808000,
        80.568000,
        81.252000,
        82.352000,
        82.112000,
        82.644000,
        83.088000,
        82.520000,
        83.788000,
        83.308000,
        84.092000,
        84.732000,
        85.164000,
        84.752000,
        84.828000,
        85.680000,
        84.852000,
        85.692000,
        85.956000,
        85.896000,
        86.116000,
        86.472000,
        86.848000,
        86.904000,
        87.008000,
        87.124000,
        87.680000,
        87.616000,
        87.328000,
        87.732000,
        88.228000,
        88.132000,
        87.836000,
        88.008000,
        87.940000,
        88.460000,
        88.588000,
        88.416000,
        88.620000,
        88.420000,
        88.644000,
        88.672000]

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# # Plot Validation Accuracy
# plt.plot(val_acc)
# plt.xlabel('Epoch')
# plt.ylabel('Validation Accuracy')
# plt.show()

dif = []
for i in range(len(val_acc)):
    if i < (len(val_acc)-1):
        dif.append(val_acc[i+1]-val_acc[i])
# print(dif)

# Show (epoch[i], beta[i-1]-beta[i])
for i, j in enumerate(dif):
    # print(i,round(j, 3))
    print(round(j,3))
    # pass

# def plot_dif(dif):
# # Plot Heatmap for differece of beta per epochs
# dif = pd.DataFrame(dif)
# dif = dif.transpose()
# plt.rcParams['figure.figsize'] = [15, 5]
# plt.pcolor(dif)
# plt.xticks(np.arange(0.5, len(dif.columns), 1), dif.columns)
# plt.yticks(np.arange(0.5, len(dif.index), 1), dif.index)
# # plt.title('Betas', fontsize=20)
# plt.xlabel('Epoch', fontsize=14)
# plt.ylabel('Operation', fontsize=14)
# plt.jet()
# plt.colorbar()
# plt.show()