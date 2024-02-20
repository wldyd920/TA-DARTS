import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import alphatolist


# path = "C:/Users/user/VSC/TD-DARTS/dragen0/alpha/"
# beta_nor, beta_red = alphatolist.beta_df(path, t=1)


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


def edge(whichdata, whichedge):
    data = whichdata.loc[whichedge].to_list()
    edge_df = pd.DataFrame(data, columns = [
                                        'none',
                                        'max_pool_3x3',
                                        'avg_pool_3x3',
                                        'skip_connect',
                                        'sep_conv_3x3',
                                        'sep_conv_5x5',
                                        'dil_conv_3x3',
                                        'dil_conv_5x5'
                                        ])
    edge_df = edge_df.transpose()
    return edge_df


def plot_edge(whichdata, whichedge):
    edge_df = edge(whichdata, whichedge)
    # plt
    plt.rcParams['figure.figsize'] = [20, 5]
    plt.pcolor(edge_df)
    plt.xticks(np.arange(0.5, len(edge_df.columns), 1), edge_df.columns)
    plt.yticks(np.arange(0.5, len(edge_df.index), 1), edge_df.index)
    # plt.title('Betas', fontsize=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Operation', fontsize=14)
    # plt.viridis()
    plt.colorbar()
    # plt.show()


def multi_edge_heatmap(whichdata, which_edges=list(range(0,14))):
    df = edge(whichdata, which_edges[0])
    for i in which_edges[1:]:
        df += edge(whichdata, i)
    plot_edge(df)
   

def save_heatmap(whichdf, nr):
    if nr == 'normal':
        createDirectory('Beta_normals')
        for i in range(0, 14):
            plot_edge(whichdf, i)
            plt.title(f'Normal {i}', fontsize=20)
            plt.savefig(f'Beta_normals/Beta_normal{i}.svg')
            plt.clf()
    elif nr == 'reduce':
        createDirectory('Beta_reduces')
        for i in range(0, 14):
            plot_edge(whichdf, i)
            plt.title(f'Reduce {i}', fontsize=20)
            plt.savefig(f'Beta_reduces/Beta_reduce{i}.svg')
            plt.clf()


###################################################################################
# test

path = "C:/Users/user/VSC/DARTS/dragen0/alpha/"
beta_nor, beta_red = alphatolist.beta_df(path, t=1)

'''
one beta_df has 50 epochs (0~49)
one epoch has 14 mixed_edges (0~13)
one mixed_edge has 8 operations (0~7)
column : epoch
row : mixed_edge no.
component : list of 8 operations' beta
'''

# # Single edge heatmap
# which_edge = 0
# df = edge(beta_nor, which_edge) 
# plot_edge(df)

# # Multi-edge heatmap
# # which_edges = [0, 1, 4, 6, 13]
# which_edges = list(range(0,14))
# print(which_edges[0])
# multi_edge_heatmap(beta_nor, which_edges)

# # All-edge heatmap
# multi_edge_heatmap(beta_nor)

# Save heatmap
save_heatmap(beta_nor, 'normal')
save_heatmap(beta_red, 'reduce')
