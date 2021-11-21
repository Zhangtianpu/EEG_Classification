import torch
import os

class Trainer(object):
    def __init__(self,model,loss,optimizer,scheduler,train_loader,val_loader,test_loader,train_config,device):
        """
        模型训练类，包含了模型训练，模型存储，模型加载以及模型评估
        :param model:
        :param loss:
        :param optimizer:
        :param scheduler:
        :param train_loader:
        :param val_loader:
        :param test_loader:
        :param train_config:
        :param device:
        """

        super(Trainer, self).__init__()
        self.model=model
        self.train_config=train_config
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.test_loader=test_loader
        self.device=device
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.loss=loss


    def train_epoch(self):
        """
        训练模型
        :return: 当前批次的损失与准确率，float,float
        """

        self.model.train()
        #当前批次模型总损失值
        total_loss=0
        #当前批次模型正确率
        correct=0.0

        #当前批次样本总数
        size = len(self.train_loader.dataset)
        for X, labels in self.train_loader:

            #将样本数据加载到gpu上
            X,labels=self._load_gpu(X,labels)

            #得到模型预测结果
            output=self.model(X)

            #计算模型损失
            loss=self.loss(output,labels)

            #计算正确率
            correct += (output.argmax(1) == labels).type(torch.float).sum().item()


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss+=loss.item()

        return total_loss/len(self.train_loader),correct/size

    def train(self):

        #最优模型正确率
        best_correct=0.0

        for epoch in range(1,self.train_config.get('epochs')+1):

            #训练模型
            train_epoch_loss,train_correct=self.train_epoch()

            #使用验证集，验证模型效果
            _,val_correct=self.evaluate(val_or_test='val')

            print("[%s]: train_loss:%.3f\ttrain_correct:%.3f\tval_correct:%.3f\tlr:%.6f" % (
            epoch, train_epoch_loss, train_correct, val_correct, self.scheduler.get_lr()[0]))

            # 获取最优模型，并将该模型使用测试集进行测试，同时保存该模型
            if best_correct<val_correct:
                best_correct=val_correct
                _, test_correct = self.evaluate(val_or_test='test')
                self.save_model(epoch)
                print("[%s]: best val_correct:%.3f\ttest_correct:%.3f" % (epoch,best_correct,test_correct))
            self.scheduler.step()


    def evaluate(self,val_or_test='val'):
        """
        验证模型的预测效果
        :param val_or_test: val加载验证数据，test加载测试数据
        :return: 验证集或着测试集的损失与准确率，float,float
        """

        self.model.eval()

        num_batches=len(self.test_loader)
        size= len(self.test_loader.dataset)
        total_loss = 0
        correct=0.0

        data_loader=self.val_loader
        if val_or_test=='tes':
            data_loader=self.test_loader

        with torch.no_grad():
            for X, labels in data_loader:
                X,labels=self._load_gpu(X, labels)
                ouptuts=self.model(X)
                total_loss+=self.loss(ouptuts,labels).item()
                correct+=(ouptuts.argmax(1)==labels).type(torch.float).sum().item()
        total_loss/=num_batches
        correct/=size
        return total_loss,correct

    def _load_gpu(self,X,labels):
        """
        加载样本数据到gpu
        :param X:
        :param labels:
        :return:
        """
        return X.float().to(self.device),labels.long().squeeze().to(self.device)

    def save_model(self,epoch):
        """
        保存模型
        :param epoch: 训练模型的批次
        :return:
        """
        save_model_path=os.path.join(self.train_config.get('experiment_dir'),'eeg_clf_%s.pkl'%epoch)
        torch.save(self.model.state_dict(),save_model_path)
        print("save model:%s"%save_model_path)

    def load_model(self,epoch):
        """
        加载指定批次的模型
        :param epoch: 模型训练批次
        :return:
        """
        load_model_path=os.path.join(self.train_config.get('experiment_dir'),'egg_clf_%s.pkl'%epoch)
        self.model.load_sate_dict(torch.load(load_model_path))
