import torch
import scipy.io as scio
import os
import pickle


class Eeg_dataset(torch.utils.data.Dataset):
    def __init__(self, data_file, train_or_test='train'):
        """
        读取数据，生成数据集
        :param data_file: 数据所在路径
        :param train_or_test: train 训练数据集，val 验证数据集，test 测试数据集
        """

        self.train_or_test = train_or_test
        self.data_file=data_file
        if not os.path.exists(self.data_file):
            raise Exception('data file does not exisit.' )

        train_test_dict=self.load_eeg_data()
        if self.train_or_test=='train':
            data=train_test_dict['train']
        else:
            data=train_test_dict['test']

        self.data_X=data['X']
        self.data_labels=data['labels']


    # 获取单条数据
    def __getitem__(self, index):
        X=self.data_X[index]
        label=self.data_labels[index]-1
        return X, label

    # 数据集长度
    def __len__(self):
        return len(self.data_labels)

    def load_eeg_data(self):
        #'sub', 'Fs', 'N', 'T', 'exemplarLabels', 'categoryLabels', 'X_2D', 'X_3D'
        with open(self.data_file,'rb') as f:
            train_test_dict=pickle.load(f)
            return train_test_dict