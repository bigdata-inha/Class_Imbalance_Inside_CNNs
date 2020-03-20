import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim
import os
from dataset import *
from trainer.trainer_vgg import *
from trainer.trainer_resnet import *


dir_10 = "C:/Users/Hongjun/Desktop/dataset/cifar10_numpy"
dir10 = "C:/Users/Hongjun/Desktop/dataset/cifar10"
dir100 = "C:/Users/Hongjun/Desktop/dataset/cifar100"

'''
train_X_dir = os.path.join(dir_10, "train_X.npy")
train_Y_dir = os.path.join(dir_10, "train_Y.npy")
test_X_dir = os.path.join(dir_10, "test_X.npy")
test_Y_dir = os.path.join(dir_10, "test_Y.npy")

train_X = np.load(train_X_dir)
train_Y = np.load(train_Y_dir)
test_X = np.load(test_X_dir)
test_Y = np.load(test_Y_dir)

trainset = [train_X, train_Y]
testset = [test_X, test_Y]

a = np.arange(0,79)
b = np.arange(80,99)
minority_label = [1,3,5,7,9]
majority_label = [0,2,4,6,8]

'''
mi = np.arange(0,50)
ma = np.arange(50,100)
#cifar10 = Cifar10(directory=dir10)
cifar100 = Cifar100(directory=dir100)
train_X, train_Y, test_X, test_Y = cifar100.initialize(directory=dir100)

balanced_trainset = [train_X, train_Y]
balanced_testset = [test_X, test_Y]


#imbalanced_cifar10 = imbalanced_Cifar10(directory=dir10, minority_label=minority_label, majority_label=majority_label, imbalance_ratio=100)
imbalanced_cifar100 = imbalanced_Cifar100(directory=dir100, minority_label=mi, majority_label=ma, imbalance_ratio=20)
imbalanced_trainset = imbalanced_cifar100.imbalance_train_set
imbalanced_testset = [test_X, test_Y]



if __name__ == "__main__" :
    #imbalanced_vgg16 = imbalanced_VGG16(trainset=imbalanced_trainset, testset=imbalanced_testset, lr=0.1, batch_norm=True)
    #imbalanced_vgg16.run(epochs=50)

    #balanced_vgg16 = balanced_VGG16(trainset=balanced_trainset, testset=balancned_testset, lr=0.1, batch_norm=True)
    #balanced_vgg16.run(epochs=50)

    balanced_res32 = balanced_Res32(trainset=balanced_trainset, testset=balanced_testset, lr=0.1)
    balanced_res32.run(epochs=50)

    imbalanced_res32 = imbalanced_Res32(trainset=imbalanced_trainset, testset=imbalanced_testset, lr=0.1)
    imbalanced_res32.run(epochs=50)










