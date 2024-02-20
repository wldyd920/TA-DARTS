import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Find directory, if not exist, make
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory.'+directory) 

# Find search acc list
def find_search_acc(path):
    f = open(path+"log.txt", 'r')
    lines = f.readlines()
    val_accs = []
    train_accs = []
    for line in lines:
        if line[24:33] == 'valid acc':
            print(line[24:33])
            val=line[35:-2]
            val_accs.append(float(val))
        elif line[24:33] == 'train acc':
            print(line[24:33])
            train=line[35:-2]
            train_accs.append(float(train))
    return val_accs, train_accs

# Find train acc list
def find_train_acc(path):
    f = open(path+"log.txt", 'r')
    lines = f.readlines()
    val_accs = []
    train_accs = []
    for line in lines:
        if line[24] == 'v' and line[30] == "a":
            val=line[35:-1]
            val_accs.append(float(val))
        elif line[24] == 't' and line[30] == "a":
            train=line[35:-1]
            train_accs.append(float(train))
    return val_accs, train_accs


def plot_search(exp_num, save=False, show=True):
    # Set path
    path = f"C:/Users/user/VSC/TD-DARTS/hist/{exp_num}/"
    file_list = os.listdir(path)
    for i in file_list:
        if i[0] == "s":
            search_path = path+f'{i}/'
    save_path = f"C:/Users/user/VSC/TD-DARTS/hist/{exp_num}/search0/"

    # Find acc
    val_accs, train_accs = find_search_acc(search_path)
    # print
    print('Search accs')
    print(f'Search val_accs: {val_accs} \n',f'Search train_accs: {train_accs}')
    print(len(val_accs))

    # Set X axis
    epochs = len(val_accs)
    epoch = list(range(epochs))

    # Plot Train, Validation Accuracy
    plt.cla()
    plt.plot(epoch, train_accs, label='train acc')
    plt.plot(epoch, val_accs, label='validation acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    if save :
        plt.savefig(save_path+'search.svg')
    if show :
        plt.show()


def plot_train(exp_num, cifar=False, save=False, show=True):
    # Set path
    path = f"C:/Users/user/VSC/TD-DARTS/hist/{exp_num}/"
    file_list = os.listdir(path)
    cnt = 0 
    if cifar == False:
        for i in file_list:
            if i[0] == "c":
                cifar_path = path+f'{i}/'
                cnt += 1
        print("Number of trainings in this search:",cnt)
        save_path = f"C:/Users/user/VSC/TD-DARTS/hist/{exp_num}/"
    else :
        cifar_path = path+f"{cifar}/"
        save_path = cifar_path
    

    # Find acc
    val_accs, train_accs = find_train_acc(cifar_path)
    # print
    print('Train accs')
    print(f'val_accs: {val_accs} \n',f'train_accs: {train_accs}')
    print(len(val_accs))

    # Set X axis
    epochs = len(val_accs)
    epoch = list(range(epochs))

    # Plot Train, Validation Accuracy
    plt.cla()
    plt.plot(epoch, train_accs, label='train acc')
    plt.plot(epoch, val_accs, label='validation acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    if save :
        plt.savefig(save_path+'train.svg')
    if show :
        plt.show()


def plot_search_comp(exp_num_list, train_or_valid, save=False, show=True):
    exp_num1, exp_num2 = exp_num_list[0], exp_num_list[1]

    # Set path
    path1 = f"C:/Users/user/VSC/TD-DARTS/hist/{exp_num1}/"
    file_list = os.listdir(path1)
    for i in file_list:
        if i[0] == "s":
            search_path1 = path1+f'{i}/'

    path2 = f"C:/Users/user/VSC/TD-DARTS/hist/{exp_num2}/"
    file_list = os.listdir(path2)
    for i in file_list:
        if i[0] == "s":
            search_path2 = path2+f'{i}/'

    save_path = f"C:/Users/user/VSC/TD-DARTS/hist/comparison/{exp_num1}_{exp_num2}/"
    createFolder(save_path)

    # Get acc list
    val_accs1, train_accs1 = find_search_acc(search_path1)
    val_accs2, train_accs2 = find_search_acc(search_path2)
    
    # Set X axis
    epochs = len(val_accs1)
    epoch = list(range(epochs))

    # Plot train, validation Accuracy
    plt.cla()

    if train_or_valid == 'train':
        plt.plot(epoch, train_accs1, label=f'{exp_num1} train acc')
        plt.plot(epoch, train_accs2, label=f'{exp_num2} train acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        if save :
            plt.savefig(save_path+'train_comparison.svg')
        if show :
            plt.show()

    elif train_or_valid == 'valid':
        plt.plot(epoch, val_accs1, label=f'{exp_num1} val acc')
        plt.plot(epoch, val_accs2, label=f'{exp_num2} val acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        if save :
            plt.savefig(save_path+'validation_comparison.svg')
        if show :
            plt.show()


def plot_train_comp(exp_num_list, train_or_valid, save=False, show=True):
    exp_num1, exp_num2 = exp_num_list[0], exp_num_list[1]

    # Set path
    path1 = f"C:/Users/user/VSC/TD-DARTS/hist/{exp_num1}/"
    file_list = os.listdir(path1)
    for i in file_list:
        if i[0] == "c":
            search_path1 = path1+f'{i}/'

    path2 = f"C:/Users/user/VSC/TD-DARTS/hist/{exp_num2}/"
    file_list = os.listdir(path2)
    for i in file_list:
        if i[0] == "c":
            search_path2 = path2+f'{i}/'

    save_path = f"C:/Users/user/VSC/TD-DARTS/hist/comparison/{exp_num1}_{exp_num2}/"
    createFolder(save_path)

    # Get acc list
    val_accs1, train_accs1 = find_train_acc(search_path1)
    val_accs2, train_accs2 = find_train_acc(search_path2)
    
    # Set X axis
    epochs = len(val_accs1)
    epoch = list(range(epochs))

    # Plot train, validation Accuracy
    plt.cla()

    if train_or_valid == 'train':
        plt.plot(epoch, train_accs1, label=f'{exp_num1} train acc')
        plt.plot(epoch, train_accs2, label=f'{exp_num2} train acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        if save :
            plt.savefig(save_path+'train_comparison.svg')
        if show :
            plt.show()

    elif train_or_valid == 'valid':
        plt.plot(epoch, val_accs1, label=f'{exp_num1} val acc')
        plt.plot(epoch, val_accs2, label=f'{exp_num2} val acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        if save :
            plt.savefig(save_path+'validation_comparison.svg')
        if show :
            plt.show()

##########################################################################################################################

# Search graph
exp_num = 69
plot_search(exp_num, save=False, show=True)

# # Train graph
# # Specify the trained model in that search number(exp_num)
# exp61_41 = "cifar10-20220908-214726"
# exp61_50 = "cifar10-20220906-212010"
# exp62_46 = "cifar100-20221205-184803"
# exp63 = "cifar100-20221205-184803"
# plot_train(exp_num, cifar=exp62_46, save=False, show=True)



##########################################################################################################################
# # Comparison graph
# exp_num_list = [59, 60]
# plot_search_comp(exp_num_list, "valid")


# # For all
# path = f"C:/Users/user/VSC/TD-DARTS/hist/"
# all_exp = os.listdir(path)
# def multiple_exp(exp_list):
#     for exp_num in exp_list:
#         plot_search(exp_num)
#         plot_train(exp_num)
#         # plot_comp(exp_num)
# # multiple_exp(all_exp)



############################################################################################################################

TDDARTS = 60

# Find Genotype list
def find_genotype(exp_numb):
    path = f"C:/Users/user/VSC/TD-DARTS/hist/{exp_numb}/search0/"
    f = open(path+"log.txt", 'r')
    lines = f.readlines()
    genotypes = []
    for line in lines:
        if len(line) < 34: pass
        elif line[34] == 'G':
            genotype=line[34:]
            genotypes.append(genotype)
    return genotypes

genotype = find_genotype(TDDARTS)
for index, i in enumerate(genotype):
    print(f'TDDARTS60_epoch{index}=', i)