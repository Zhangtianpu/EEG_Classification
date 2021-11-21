import argparse
import yaml
import eeg_dataset
from  torch.utils.data.dataloader import DataLoader
import basic_trainer
import torch.optim as optim
from eeg_network import eeg_model
import torch.nn as nn
import torch
import shutil
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(config_path):
    """
    加载配置文件，训练模型
    :param config_path: 配置文件路径
    :return:
    """
    with open(config_path) as f:
        """
        记载配置文件
        """
        config_yaml=yaml.load(f)
        data_config=config_yaml['data']
        train_config=config_yaml['train']
        model_config=config_yaml['model']

        train_test_data_file=data_config.get('train_test_data_file')
        batch_size=train_config.get('batch_size')
        experiment_dir=train_config.get('experiment_dir')
        steps=train_config.get('steps')
        lr_decay_ratio=train_config.get('lr_decay_ratio')

        """
        创建模型保存文件夹
        """
        if os.path.exists(experiment_dir):
            shutil.rmtree(experiment_dir)
        os.makedirs(experiment_dir)
        print('create experiment dir : %s'%experiment_dir)


        """
        生成训练、验证、测试数据集
        """
        train_data = eeg_dataset.Eeg_dataset(train_test_data_file,'train')
        val_data=eeg_dataset.Eeg_dataset(train_test_data_file,'val')
        test_data=eeg_dataset.Eeg_dataset(train_test_data_file,'test')

        train_loader = DataLoader(dataset=train_data, batch_size=batch_size,shuffle=True)
        val_loader= DataLoader(dataset=val_data, batch_size=batch_size,shuffle=True)
        test_loader=DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

        """
        创建模型
        """
        model=eeg_model(model_config,device).to(device)

        """
        设置损失函数，优化算法
        """
        loss=nn.CrossEntropyLoss()

        optimizer=optim.Adam(params=model.parameters(), lr=train_config.get('base_lr'), eps=1.0e-8)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,gamma=lr_decay_ratio)
        # optimizer = optim.SGD(params=model.parameters(), lr=train_config.get('base_lr'),  momentum=0.9)

        """
        模型训练
        """
        eeg_trainer=basic_trainer.Trainer(model,loss,optimizer,lr_scheduler,train_loader,val_loader,test_loader,train_config,device)
        eeg_trainer.train()


if __name__ == '__main__':
    """
    设置配置文件路径
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configurations/eeg_config.yaml')
    args = parser.parse_args()


    main(args.config_path)