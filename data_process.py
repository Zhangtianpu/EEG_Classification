import scipy.io as scio
import argparse
import yaml
import os
import numpy as np
import pickle

def save_data(data,path):
    """
    保存数据
    :param data:
    :param path:
    :return:
    """
    with open(path,'wb') as f:
        pickle.dump(data,f)

def train_test_split(X,y,train_prop=0.6,val_prop=0.2):
    """
    划分数据集
    :param X:
    :param y:
    :param train_prop: 训练数据集所占比例
    :param val_prop: 验证数据集所占比例
    :return:
    """
    X=np.array(X)
    y=np.array(y)
    num_of_samples=X.shape[0]

    #打乱数据分布
    random_index=np.random.permutation(num_of_samples)
    random_X=X[random_index]
    random_y=y[random_index]

    #generate train data
    train_index=int(num_of_samples*train_prop)
    train_X=random_X[:train_index]
    train_y=random_y[:train_index]

    #generate val data
    val_index=int(num_of_samples*val_prop)
    val_X = random_X[train_index:train_index+val_index]
    val_y = random_y[train_index:train_index+val_index]

    #generate test data
    test_X = random_X[train_index+val_index:]
    test_y = random_y[train_index+val_index:]


    return train_X,train_y,val_X,val_y,test_X,test_y

def load_data(path):
    """
    读取.mat的原始eeg数据
    :param path:
    :return:
    """
    data=scio.loadmat(path)
    labels = data['categoryLabels'].transpose(1, 0)
    X = data['X_3D'].transpose(2, 1, 0)
    return X,labels

def main(config_path):
    with open(config_path) as f:
        data_dict = {"train": {}, "val":{},"test": {}}
        total_X=[]
        total_labels=[]
        config_yaml=yaml.load(f)
        data_config=config_yaml['data']
        raw_data_dir=data_config.get('raw_data_dir')
        train_test_data_file=data_config.get('train_test_data_file')
        if not os.path.exists(raw_data_dir):
            raise Exception('raw data dir does not exisit.' )

        #读取原始数据
        file_names=os.listdir(raw_data_dir)
        file_path_list=[os.path.join(raw_data_dir,file_name) for file_name in file_names]
        for file_path in file_path_list:
            X,labels=load_data(file_path)
            total_X.extend(X)
            total_labels.extend(labels)

        #划分训练，验证，测试数据集
        train_X,train_y,val_X,val_y,test_X,test_y=train_test_split(total_X,total_labels)


        data_dict['train']['X']=train_X
        data_dict['train']['labels'] = train_y
        data_dict['val']['X'] = val_X
        data_dict['val']['labels'] = val_y
        data_dict['test']['X'] = test_X
        data_dict['test']['labels'] = test_y

        save_data(data_dict,train_test_data_file)

if __name__ == '__main__':
    """
    指定配置文件
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configurations/eeg_config.yaml')
    args=parser.parse_args()

    main(args.config_path)