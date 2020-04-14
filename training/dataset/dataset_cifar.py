import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
from training.dataset.data_preprocessing import *

def set_seed(seed):
    """
    Set a random seed for numpy and PyTorch.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)



dir_10 = "C:/Users/Hongjun/Desktop/dataset/cifar10"
dir_100 = "C:/Users/Hongjun/Desktop/dataset/cifar100"


class Cifar10() :
    def __init__(self, directory=None, step = None, number_label = None):
        self.directory = directory
        self.step = step
        self.number_label = number_label
        self.train_X, self.train_Y, self.test_X, self.test_Y = self.initialize(directory=self.directory)
        self.ordered_test_X, self.ordered_test_Y = self.init_ordered_testset(self.test_X, self.test_Y)

    def load_cifar10_batch(self, directory):
        """ load batch of cifar """
        with open(directory, 'rb') as fo:
            datadict = pickle.load(fo, encoding='bytes')
        X = np.array(datadict[b'data'])
        Y = np.array(datadict[b'labels'])
        return X, Y

    def load_cifar10(self, directory):
        """ load all of cifar10 """
        train_data = []
        train_labels = []
        for b in range(1, 6):
            f = os.path.join(directory, 'data_batch_%d' % (b,))
            X, Y = self.load_cifar10_batch(f)
            train_data.append(X)
            train_labels.append(Y)
        train_data = np.concatenate(train_data)
        train_labels = np.concatenate(train_labels)
        del X, Y
        test_data, test_labels = self.load_cifar10_batch(os.path.join(directory, 'test_batch'))
        return train_data, train_labels, test_data, test_labels

    def load_cifar10_img_form(self, directory):
        """
        cifar10 데이터를 이미지 형태로 불러옵니다
        :param directory: cifar10 batch 파일 위치
        :return:
            N, H, W, C 순서의 cifar 10 데이터
        """
        train_data, train_labels, test_data, test_labels = self.load_cifar10(directory)

        R, testR = train_data[:, :1024].reshape(-1, 32, 32, 1), test_data[:, :1024].reshape(-1, 32, 32, 1)
        G, testG = train_data[:, 1024:2048].reshape(-1, 32, 32, 1), test_data[:, 1024:2048].reshape(-1, 32, 32, 1)
        B, testB = train_data[:, 2048:].reshape(-1, 32, 32, 1), test_data[:, 2048:].reshape(-1, 32, 32, 1)

        '''
        R, testR = train_data[:, :1024].reshape(-1, 1, 32, 32), test_data[:, :1024].reshape(-1, 1, 32, 32)
        G, testG = train_data[:, 1024:2048].reshape(-1, 1, 32, 32), test_data[:, 1024:2048].reshape(-1, 1, 32, 32)
        B, testB = train_data[:, 2048:].reshape(-1, 1, 32, 32), test_data[:, 2048:].reshape(-1, 1, 32, 32)
        '''

        train_data, test_data = np.concatenate((R, G, B), axis=3), np.concatenate((testR, testG, testB), axis=3)
        return train_data, train_labels, test_data, test_labels

    def initialize(self, directory) :
        train_data, train_labels, test_data, test_labels = self.load_cifar10_img_form(directory=directory)
        train_labels, test_labels = torch.as_tensor(train_labels, dtype=torch.long), torch.as_tensor(test_labels,
                                                                                                     dtype=torch.long)
        return train_data, train_labels, test_data, test_labels

    def init_ordered_testset(self, data_X, data_Y):

        temp_list = self._split_by_label(10, data_X, data_Y)
        temp_X = []
        temp_Y = []
        for i in range(temp_list.__len__()) :
            temp_X.append(temp_list[i][0])
            temp_Y.append(temp_list[i][1])

        test_X = np.concatenate(temp_X, axis = 0)
        test_Y = np.concatenate(temp_Y, axis = 0)

        return test_X, test_Y

    def _split_by_label(self, n_label, data_X, data_Y):
        '''load 된 데이터 label별로 나누는 함수
        '''
        list = []
        for k in range(n_label):
            # list[k] = []
            k_label = np.where(data_Y == k)
            data = data_X[k_label]
            label = data_Y[k_label]
            # data = torch.as_tensor(data_X[k_label],dtype = torch.float)
            # label = torch.as_tensor(data_Y[k_label], dtype= torch.long)
            new_list = [data, label]
            list.append(new_list)

        return list

class imbalanced_Cifar10() :
    def __init__(self, directory=None, step = None, minority_label = None, majority_label=None, imbalance_ratio = None):
        self.directory = directory
        self.step = step
        self.minority_label = minority_label
        self.majority_label = majority_label
        self.imbalance_ratio = imbalance_ratio

        self.train_X, self.train_Y, self.test_X, self.test_Y = self.initialize(directory=self.directory)

        self.split_dataset= self._split_by_label(10, self.train_X, self.train_Y)

        self.imbalance_train_set = self.make_imbalance_dataset(split_dataset = self.split_dataset, majority_label=self.majority_label,
                                                               minority_label = self.minority_label)

    def load_cifar10_batch(self, directory):
        """ load batch of cifar """
        with open(directory, 'rb') as fo:
            datadict = pickle.load(fo, encoding='bytes')
        X = np.array(datadict[b'data'])
        Y = np.array(datadict[b'labels'])
        return X, Y

    def load_cifar10(self, directory):
        """ load all of cifar10 """
        train_data = []
        train_labels = []
        for b in range(1, 6):
            f = os.path.join(directory, 'data_batch_%d' % (b,))
            X, Y = self.load_cifar10_batch(f)
            train_data.append(X)
            train_labels.append(Y)
        train_data = np.concatenate(train_data)
        train_labels = np.concatenate(train_labels)
        del X, Y
        test_data, test_labels = self.load_cifar10_batch(os.path.join(directory, 'test_batch'))
        return train_data, train_labels, test_data, test_labels

    def load_cifar10_img_form(self, directory):
        """
        cifar10 데이터를 이미지 형태로 불러옵니다
        :param directory: cifar10 batch 파일 위치
        :return:
            N, H, W, C 순서의 cifar 10 데이터
        """
        train_data, train_labels, test_data, test_labels = self.load_cifar10(directory)

        R, testR = train_data[:, :1024].reshape(-1, 32, 32, 1), test_data[:, :1024].reshape(-1, 32, 32, 1)
        G, testG = train_data[:, 1024:2048].reshape(-1, 32, 32, 1), test_data[:, 1024:2048].reshape(-1, 32, 32, 1)
        B, testB = train_data[:, 2048:].reshape(-1, 32, 32, 1), test_data[:, 2048:].reshape(-1, 32, 32, 1)

        '''
        R, testR = train_data[:, :1024].reshape(-1, 1, 32, 32), test_data[:, :1024].reshape(-1, 1, 32, 32)
        G, testG = train_data[:, 1024:2048].reshape(-1, 1, 32, 32), test_data[:, 1024:2048].reshape(-1, 1, 32, 32)
        B, testB = train_data[:, 2048:].reshape(-1, 1, 32, 32), test_data[:, 2048:].reshape(-1, 1, 32, 32)
        '''

        train_data, test_data = np.concatenate((R, G, B), axis=3), np.concatenate((testR, testG, testB), axis=3)
        return train_data, train_labels, test_data, test_labels

    def initialize(self, directory) :
        train_data, train_labels, test_data, test_labels = self.load_cifar10_img_form(directory=directory)
        train_labels, test_labels = torch.as_tensor(train_labels, dtype=torch.long), torch.as_tensor(test_labels,
                                                                                                     dtype=torch.long)
        return train_data, train_labels, test_data, test_labels

    def _split_by_label(self, n_label, data_X, data_Y):
        '''load 된 데이터 label별로 나누는 함수
        '''
        list = []
        for k in range(n_label):
            # list[k] = []
            k_label = np.where(data_Y == k)
            data = data_X[k_label]
            label = data_Y[k_label]
            # data = torch.as_tensor(data_X[k_label],dtype = torch.float)
            # label = torch.as_tensor(data_Y[k_label], dtype= torch.long)
            new_list = [data, label]
            list.append(new_list)

        return list

    def make_imbalance_dataset(self, split_dataset, majority_label, minority_label):
        '''

        :param split_dataset: dataset which is spliited by label
        :return: imbalanced dataset(dtype : list)
        '''

        temp_data_X = []
        temp_data_Y = []
        for i in majority_label :
            if i == 10 :
                break
            #temp_label = np.where(split_dataset[current_step][1] == i)
            # np.where 함수 output array형태로 두개 나옴, [0]에 index가 tuple형태

            temp_data_X.append(split_dataset[i][0])
            temp_data_Y.append(split_dataset[i][1])

        if i==10 :
            majority_data_X = None
            majority_data_Y = None
        else :
            majority_data_X = np.concatenate(temp_data_X, axis=0)
            majority_data_Y = np.concatenate(temp_data_Y, axis=0)

        del temp_data_X, temp_data_Y

        temp_data_X = []
        temp_data_Y = []

        for j in minority_label :
            temp_index = np.random.choice(np.arange(0, len(split_dataset[j][1])), int(len(split_dataset[j][1])/self.imbalance_ratio),
                                          replace=False)

            #np.where 함수 output array형태로 두개 나옴, [0]에 index가 tuple형태

            temp_data_X.append(split_dataset[j][0][temp_index])
            temp_data_Y.append(split_dataset[j][1][temp_index])

        minority_data_X = np.concatenate(temp_data_X, axis=0)
        minority_data_Y = np.concatenate(temp_data_Y, axis=0)

        if i==10 :
            imbalanced_data_X = minority_data_X
            imbalanced_data_Y = minority_data_Y
        else :
            imbalanced_data_X = np.concatenate((minority_data_X, majority_data_X), axis=0)
            imbalanced_data_Y = np.concatenate((minority_data_Y, majority_data_Y), axis=0)

        imbalanced_dataset =[imbalanced_data_X, imbalanced_data_Y]

        del temp_data_X, temp_data_Y
        return imbalanced_dataset


class Cifar100() :
    def __init__(self, directory=None, minority_label=None, majority_label=None, imbalance_ratio=None):
        self.directory = directory
        self.minority_label = minority_label
        self.majority_label = majority_label
        self.imbalance_ratio = imbalance_ratio

        self.train_X, self.train_Y, self.test_X, self.test_Y = self.initialize(directory=self.directory)
        self.ordered_test_X, self.ordered_test_Y = self.init_ordered_testset(self.test_X, self.test_Y)


    def load_cifar100_batch(self, directory):
        """ load batch of cifar """
        print(directory)
        with open(directory, 'rb') as fo:
            datadict = pickle.load(fo, encoding='bytes')
        X = np.array(datadict[b'data'])
        Y = np.array(datadict[b'fine_labels'])
        return X, Y

    def load_cifar100(self, directory):
        """ load all of cifar100 """
        train_data = []
        train_labels = []

        f = os.path.join(directory, 'train')

        X, Y = self.load_cifar100_batch(f)
        train_data.append(X)
        train_labels.append(Y)
        train_data = np.concatenate(train_data)
        train_labels = np.concatenate(train_labels)
        del X, Y
        test_data, test_labels = self.load_cifar100_batch(os.path.join(directory, 'test'))

        return train_data, train_labels, test_data, test_labels


    def load_cifar100_img_form(self, directory):
        """
        cifar10 데이터를 이미지 형태로 불러옵니다
        :param directory: cifar100 train / test 파일 위치
        :return:
            N, H, W, C 순서의 cifar 10 데이터
        """
        train_data, train_labels, test_data, test_labels = self.load_cifar100(directory)

        R, testR = train_data[:, :1024].reshape(-1, 32, 32, 1), test_data[:, :1024].reshape(-1, 32, 32, 1)
        G, testG = train_data[:, 1024:2048].reshape(-1, 32, 32, 1), test_data[:, 1024:2048].reshape(-1, 32, 32, 1)
        B, testB = train_data[:, 2048:].reshape(-1, 32, 32, 1), test_data[:, 2048:].reshape(-1, 32, 32, 1)

        '''
        R, testR = train_data[:, :1024].reshape(-1, 1, 32, 32), test_data[:, :1024].reshape(-1, 1, 32, 32)
        G, testG = train_data[:, 1024:2048].reshape(-1, 1, 32, 32), test_data[:, 1024:2048].reshape(-1, 1, 32, 32)
        B, testB = train_data[:, 2048:].reshape(-1, 1, 32, 32), test_data[:, 2048:].reshape(-1, 1, 32, 32)
        '''
        train_data, test_data = np.concatenate((R, G, B), axis=3), np.concatenate((testR, testG, testB), axis=3)
        return train_data, train_labels, test_data, test_labels


    def _split_by_label(self, n_label, data_X, data_Y):
        '''load 된 데이터 label별로 나누는 함수
        '''
        list = []
        for k in range(n_label):
            # list[k] = []
            k_label = np.where(data_Y == k)
            data = data_X[k_label]
            label = data_Y[k_label]
            # data = torch.as_tensor(data_X[k_label],dtype = torch.float)
            # label = torch.as_tensor(data_Y[k_label], dtype= torch.long)
            new_list = [data, label]
            list.append(new_list)

        return list

    def initialize(self, directory) :
        train_data, train_labels, test_data, test_labels = self.load_cifar100_img_form(directory=directory)
        train_labels, test_labels = torch.as_tensor(train_labels, dtype=torch.long), torch.as_tensor(test_labels,
                                                                                                     dtype=torch.long)
        return train_data, train_labels, test_data, test_labels

    def init_ordered_testset(self, data_X, data_Y):

        temp_list = self._split_by_label(100, data_X, data_Y)
        temp_X = []
        temp_Y = []
        for i in range(temp_list.__len__()) :
            temp_X.append(temp_list[i][0])
            temp_Y.append(temp_list[i][1])

        test_X = np.concatenate(temp_X, axis = 0)
        test_Y = np.concatenate(temp_Y, axis = 0)

        return test_X, test_Y


class imbalanced_Cifar100() :
    def __init__(self, directory=None, minority_label=None, majority_label=None, imbalance_ratio=None):
        set_seed(2020)
        self.directory = directory
        self.minority_label = minority_label
        self.majority_label = majority_label
        self.imbalance_ratio = imbalance_ratio

        self.train_X, self.train_Y, self.test_X, self.test_Y = self.initialize(directory=self.directory)
        self.split_dataset = self._split_by_label(100, self.train_X, self.train_Y)

        self.imbalance_train_set = self.make_imbalance_dataset(split_dataset=self.split_dataset,
                                                               majority_label=self.majority_label,
                                                               minority_label=self.minority_label)

    def load_cifar100_batch(self, directory):
        """ load batch of cifar """
        with open(directory, 'rb') as fo:
            datadict = pickle.load(fo, encoding='bytes')
        X = np.array(datadict[b'data'])
        Y = np.array(datadict[b'fine_labels'])
        return X, Y

    def load_cifar100(self, directory):
        """ load all of cifar100 """
        train_data = []
        train_labels = []

        f = os.path.join(directory, 'train')

        X, Y = self.load_cifar100_batch(f)
        train_data.append(X)
        train_labels.append(Y)
        train_data = np.concatenate(train_data)
        train_labels = np.concatenate(train_labels)
        del X, Y
        test_data, test_labels = self.load_cifar100_batch(os.path.join(directory, 'test'))

        return train_data, train_labels, test_data, test_labels


    def load_cifar100_img_form(self, directory):
        """
        cifar10 데이터를 이미지 형태로 불러옵니다
        :param directory: cifar100 train / test 파일 위치
        :return:
            N, H, W, C 순서의 cifar 10 데이터
        """
        train_data, train_labels, test_data, test_labels = self.load_cifar100(directory)

        R, testR = train_data[:, :1024].reshape(-1, 32, 32, 1), test_data[:, :1024].reshape(-1, 32, 32, 1)
        G, testG = train_data[:, 1024:2048].reshape(-1, 32, 32, 1), test_data[:, 1024:2048].reshape(-1, 32, 32, 1)
        B, testB = train_data[:, 2048:].reshape(-1, 32, 32, 1), test_data[:, 2048:].reshape(-1, 32, 32, 1)

        '''
        R, testR = train_data[:, :1024].reshape(-1, 1, 32, 32), test_data[:, :1024].reshape(-1, 1, 32, 32)
        G, testG = train_data[:, 1024:2048].reshape(-1, 1, 32, 32), test_data[:, 1024:2048].reshape(-1, 1, 32, 32)
        B, testB = train_data[:, 2048:].reshape(-1, 1, 32, 32), test_data[:, 2048:].reshape(-1, 1, 32, 32)
        '''
        train_data, test_data = np.concatenate((R, G, B), axis=3), np.concatenate((testR, testG, testB), axis=3)
        return train_data, train_labels, test_data, test_labels


    def _split_by_label(self, n_label, data_X, data_Y):
        '''load 된 데이터 label별로 나누는 함수
        '''
        list = []
        for k in range(n_label):
            # list[k] = []
            k_label = np.where(data_Y == k)
            data = data_X[k_label]
            label = data_Y[k_label]
            # data = torch.as_tensor(data_X[k_label],dtype = torch.float)
            # label = torch.as_tensor(data_Y[k_label], dtype= torch.long)
            new_list = [data, label]
            list.append(new_list)

        return list

    def initialize(self, directory) :
        train_data, train_labels, test_data, test_labels = self.load_cifar100_img_form(directory=directory)
        train_labels, test_labels = torch.as_tensor(train_labels, dtype=torch.long), torch.as_tensor(test_labels,
                                                                                                     dtype=torch.long)
        return train_data, train_labels, test_data, test_labels

    def make_imbalance_dataset(self, split_dataset, majority_label, minority_label):
        '''

        :param split_dataset: dataset which is spliited by label
        :return: imbalanced dataset(dtype : list)
        '''

        temp_data_X = []
        temp_data_Y = []
        for i in majority_label :

            #temp_label = np.where(split_dataset[current_step][1] == i)
            # np.where 함수 output array형태로 두개 나옴, [0]에 index가 tuple형태

            temp_data_X.append(split_dataset[i][0])
            temp_data_Y.append(split_dataset[i][1])

        majority_data_X = np.concatenate(temp_data_X, axis=0)
        majority_data_Y = np.concatenate(temp_data_Y, axis=0)

        del temp_data_X, temp_data_Y

        temp_data_X = []
        temp_data_Y = []

        for j in minority_label :
            set_seed(2020)
            temp_index = np.random.choice(np.arange(0, len(split_dataset[i][1])), int(len(split_dataset[i][1])/self.imbalance_ratio),
                                          replace=False)

            #np.where 함수 output array형태로 두개 나옴, [0]에 index가 tuple형태

            temp_data_X.append(split_dataset[j][0][temp_index])
            temp_data_Y.append(split_dataset[j][1][temp_index])

        minority_data_X = np.concatenate(temp_data_X, axis=0)
        minority_data_Y = np.concatenate(temp_data_Y, axis=0)

        imbalanced_data_X = np.concatenate((minority_data_X, majority_data_X), axis=0)
        imbalanced_data_Y = np.concatenate((minority_data_Y, majority_data_Y), axis=0)

        imbalanced_dataset =[imbalanced_data_X, imbalanced_data_Y]

        del temp_data_X, temp_data_Y
        return imbalanced_dataset



