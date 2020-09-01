import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim
import os
from training.dataset import *
from training.trainer.trainer_vgg import *
from training.trainer.trainer_resnet import *
from training.dataset.dataset_cifar import *
from training.dataset.dataset_imagenet import *



dir_10 = "C:/Users/Hongjun/Desktop/dataset/cifar10_numpy"
dir10 = "C:/Users/Hongjun/Desktop/dataset/cifar10"
dir100 = "C:/Users/Hongjun/Desktop/dataset/cifar100"


set_seed(2020)
mi = np.arange(0,50)
ma = np.arange(50,100)

mi_10 = np.arange(0,5)
ma_10 = np.arange(5,10)


cifar10 = Cifar10(directory=dir10)
train_X, train_Y, test_X, test_Y = cifar10.initialize(directory=dir10)
test_X = cifar10.ordered_test_X
test_Y = cifar10.ordered_test_Y

'''
cifar100 = Cifar100(directory=dir100)
train_X, train_Y, test_X, test_Y = cifar100.initialize(directory=dir100)
test_X = cifar100.ordered_test_X
test_Y = cifar100.ordered_test_Y
'''

balanced_trainset = [train_X, train_Y]
balanced_testset = [test_X, test_Y]

'''Cifar 100'''
'''
if __name__ == "__main__" :
    
    ####################################################vgg########################################################

    #vgg imbalance ratio : 20
    imbalanced_cifar100 = imbalanced_Cifar100(directory=dir100, minority_label=mi, majority_label=ma,
                                              imbalance_ratio=20)

    imbalanced_trainset = imbalanced_cifar100.imbalance_train_set
    imbalanced_testset = [test_X, test_Y]

    imbalanced_vgg16 = imbalanced_VGG16(trainset=imbalanced_trainset, testset=imbalanced_testset, lr=0.1,
                                        batch_norm=True, imbalance_ratio=20, mu=0.5)
    imbalanced_vgg16.run(epochs=100)

    #vgg imbalance ratio : 50
    imbalanced_cifar100 = imbalanced_Cifar100(directory=dir100, minority_label=mi, majority_label=ma,
                                              imbalance_ratio=50)
    imbalanced_trainset = imbalanced_cifar100.imbalance_train_set
    imbalanced_testset = [test_X, test_Y]

    imbalanced_vgg16 = imbalanced_VGG16(trainset=imbalanced_trainset, testset=imbalanced_testset, lr=0.1,
                                        batch_norm=True, imbalance_ratio=50, mu=0.5)
    imbalanced_vgg16.run(epochs=100)

    #vgg imbalance ratio : 100
    imbalanced_cifar100 = imbalanced_Cifar100(directory=dir100, minority_label=mi, majority_label=ma,
                                              imbalance_ratio=100)
    imbalanced_trainset = imbalanced_cifar100.imbalance_train_set
    imbalanced_testset = [test_X, test_Y]

    imbalanced_vgg16 = imbalanced_VGG16(trainset=imbalanced_trainset, testset=imbalanced_testset, lr=0.1,
                                        batch_norm=True, imbalance_ratio=100, mu=0.5)
    imbalanced_vgg16.run(epochs=100)
        
    #balanced vgg
    balanced_vgg16 = balanced_VGG16(trainset=balanced_trainset, testset=balanced_testset, lr=0.1, batch_norm=True)
    balanced_vgg16.run(epochs=100)

    del imbalanced_vgg16, balanced_vgg16


####################################################resnet########################################################

    imbalanced_cifar100 = imbalanced_Cifar100(directory=dir100, minority_label=mi, majority_label=ma,
                                              imbalance_ratio=20)

    imbalanced_trainset = imbalanced_cifar100.imbalance_train_set
    imbalanced_testset = [test_X, test_Y]

    imbalanced_res32 = imbalanced_Res32(trainset=imbalanced_trainset, testset=imbalanced_testset, lr=0.1,
                                        imbalance_ratio=20, mu=0.5)
    imbalanced_res32.run(epochs=100)


    #vgg imbalance ratio : 50
    imbalanced_cifar100 = imbalanced_Cifar100(directory=dir100, minority_label=mi, majority_label=ma,
                                              imbalance_ratio=50)

    imbalanced_trainset = imbalanced_cifar100.imbalance_train_set
    imbalanced_testset = [test_X, test_Y]

    imbalanced_res32 = imbalanced_Res32(trainset=imbalanced_trainset, testset=imbalanced_testset, lr=0.1,
                                        imbalance_ratio=50, mu=0.5)
    imbalanced_res32.run(epochs=100)

    #vgg imbalance ratio : 100
    imbalanced_cifar100 = imbalanced_Cifar100(directory=dir100, minority_label=mi, majority_label=ma,
                                              imbalance_ratio=100)
    imbalanced_trainset = imbalanced_cifar100.imbalance_train_set
    imbalanced_testset = [test_X, test_Y]

    imbalanced_res32 = imbalanced_Res32(trainset=imbalanced_trainset, testset=imbalanced_testset, lr=0.1,
                                        imbalance_ratio=100, mu=0.5)
    imbalanced_res32.run(epochs=100)

    #balanced vgg
    balanced_res32 = balanced_Res32(trainset=balanced_trainset, testset=balanced_testset, lr=0.1)
    balanced_res32.run(epochs=100)
'''

if __name__ == "__main__":

    imbalanced_cifar10 = imbalanced_Cifar10(directory=dir10, minority_label=mi_10, majority_label=ma_10,
                                        imbalance_ratio=20)

    imbalanced_trainset = imbalanced_cifar10.imbalance_train_set
    imbalanced_testset = [test_X, test_Y]

    imbalanced_vgg16 = imbalanced_VGG16_10(trainset=imbalanced_trainset, testset=imbalanced_testset, lr=0.1,
                                       batch_norm=True, imbalance_ratio=20, mu=0.5)
    imbalanced_vgg16.run(epochs=100)

    imbalanced_cifar10 = imbalanced_Cifar10(directory=dir10, minority_label=mi_10, majority_label=ma_10,
                                          imbalance_ratio=50)
    imbalanced_trainset = imbalanced_cifar10.imbalance_train_set
    imbalanced_testset = [test_X, test_Y]

    imbalanced_vgg16 = imbalanced_VGG16_10(trainset=imbalanced_trainset, testset=imbalanced_testset, lr=0.1,
                                    batch_norm=True, imbalance_ratio=50, mu=0.5)
    imbalanced_vgg16.run(epochs=100)

    imbalanced_cifar10 = imbalanced_Cifar10(directory=dir10, minority_label=mi_10, majority_label=ma_10,
                                        imbalance_ratio=100)
    imbalanced_trainset = imbalanced_cifar10.imbalance_train_set
    imbalanced_testset = [test_X, test_Y]

    imbalanced_vgg16 = imbalanced_VGG16_10(trainset=imbalanced_trainset, testset=imbalanced_testset, lr=0.1,
                                       batch_norm=True, imbalance_ratio=100, mu=0.5)
    imbalanced_vgg16.run(epochs=100)

# balanced vgg
    balanced_vgg16 = balanced_VGG16_10(trainset=balanced_trainset, testset=balanced_testset, lr=0.1, batch_norm=True)
    balanced_vgg16.run(epochs=100)

    del imbalanced_vgg16, balanced_vgg16

''''
####################################################resnet########################################################

    imbalanced_cifar10 = imbalanced_Cifar10(directory=dir10, minority_label=mi_10, majority_label=ma_10,
                                          imbalance_ratio=20)
    imbalanced_trainset = imbalanced_cifar10.imbalance_train_set
    imbalanced_testset = [test_X, test_Y]

    imbalanced_res32 = imbalanced_Res32_10(trainset=imbalanced_trainset, testset=imbalanced_testset, lr=0.1,
                                    imbalance_ratio=20, mu=0.5)
    imbalanced_res32.run(epochs=100)

# vgg imbalance ratio : 50
    imbalanced_cifar10 = imbalanced_Cifar10(directory=dir10, minority_label=mi_10, majority_label=ma_10,
                                          imbalance_ratio=50)

    imbalanced_trainset = imbalanced_cifar10.imbalance_train_set
    imbalanced_testset = [test_X, test_Y]

    imbalanced_res32 = imbalanced_Res32_10(trainset=imbalanced_trainset, testset=imbalanced_testset, lr=0.1,
                                    imbalance_ratio=50, mu=0.5)
    imbalanced_res32.run(epochs=100)

# vgg imbalance ratio : 100
    imbalanced_cifar10 = imbalanced_Cifar10(directory=dir10, minority_label=mi_10, majority_label=ma_10,
                                          imbalance_ratio=100)

    imbalanced_trainset = imbalanced_cifar10.imbalance_train_set
    imbalanced_testset = [test_X, test_Y]

    imbalanced_res32 = imbalanced_Res32_10(trainset=imbalanced_trainset, testset=imbalanced_testset, lr=0.1,
                                    imbalance_ratio=100, mu=0.5)
    imbalanced_res32.run(epochs=100)

# balanced res32
    balanced_res32 = balanced_Res32_10(trainset=balanced_trainset, testset=balanced_testset, lr=0.1)
    balanced_res32.run(epochs=100)
'''