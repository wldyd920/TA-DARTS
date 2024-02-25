import os
import wget
import torchvision.datasets as dset
import torch

imagenet_data = dset.ImageNet('C:/Users/user/VSC/TD-DARTS/data/imagenet')
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=128,
                                          shuffle=True,
                                          num_workers=1)