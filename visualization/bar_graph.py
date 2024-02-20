from softmaxfunc import softmax_temp
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

alpha = [0.2, 0.25, 0.4, 0.15]
beta = softmax_temp(alpha, t=0.1)


#plt
x_name = ['1', '2', '3', '4']
y_name = np.arange(0,1.1,0.1)


plt.subplot(1,2,1)
plt.title('Alpha', fontsize=18)
plt.xlabel('Operations', fontsize=16)
plt.bar(x_name, alpha)
plt.xticks(x_name, fontsize=12)
plt.yticks(y_name, fontsize=12)


plt.subplot(1,2,2)
plt.title('Beta', fontsize=18)
plt.xlabel('Operations', fontsize=16)
plt.bar(x_name, beta)
plt.xticks(x_name, fontsize=12)
plt.yticks(y_name, fontsize=12)

plt.subplots_adjust(wspace=0.5)

plt.show()
