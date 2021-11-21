import torch.nn as nn
import torch.nn.functional as F
import torch

class gcn_model(nn.Module):
    """
    图卷积模块
    """
    def __init__(self,num_nodes,gcn_output_dim,input_dim):
        super(gcn_model, self).__init__()
        self.num_nodes=num_nodes
        self.input_dim=input_dim
        self.gcn_output_dim=gcn_output_dim

        self.linear = nn.Linear(self.num_nodes * self.input_dim, self.num_nodes*self.gcn_output_dim )



    def forward(self,inputs,embding_node):
        """
        图卷积公式：G(X)=AXW+b，A为邻接矩阵，X输入特征，W、b可学习参数
        :param inputs: the shape of input:[batch,nodes,out_channel]
        :param embding_node: [nodes,embding_dim]
        :return:
        """
        #由于该数据集没有图结构，通过embedding的方式创建邻接矩阵，通过模型学习构造该数据集的邻接矩阵
        # the shape of adpative_adj:[nodes,nodes]
        adpative_adj=F.softmax(F.relu(torch.mm(embding_node,embding_node.transpose(0,1))),dim=1)

        # the shape of inputs_g:[batch,nodes,out_channel]
        inputs_g=torch.einsum("nm,bmf->bnf", adpative_adj,inputs)

        # the shape of inputs_g:[batch,nodes*out_channel]
        inputs_g=torch.reshape(inputs_g,(-1,self.num_nodes*self.input_dim))

        # the shape of inputs_g:[batch,nodes,spatial_channel]
        inputs_g=self.linear(inputs_g).reshape(-1,self.num_nodes,self.gcn_output_dim)
        return inputs_g


class Temporal_Gated_conv(nn.Module):
    """
    时序卷积模块，通过一位卷积提取时序关系
    """
    def __init__(self,in_channels,out_channels,kernel_size,padding=0,stride=1):
        super(Temporal_Gated_conv, self).__init__()
        #一维卷积
        self.conv_1=nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride
                              )
        self.conv_2=nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride)
        self.conv_3=nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride)


    def forward(self,X):
        """

        :param X: input size of X is [batch_size,num_node,in_channel]
        :return: output size is [batch_size,num_node,out_channel]
        """
        #after permute, X:[batch_size,input_dim,num_node]
        X=X.permute(0,2,1)

        #sig_re:[batch_size,out_channel,num_node]
        sig_re=torch.sigmoid(self.conv_2(X))

        # GLU_result:[batch_size,out_channel,num_node]
        GLU_result=self.conv_1(X).mul(sig_re)

        # conv_x_3:[batch_size,out_channel,num_node]
        conv_x_3=self.conv_3(X)

        # temporal_result:[batch_size,out_channel,num_node]
        temporal_result=GLU_result+conv_x_3

        #after permute, temporal_result:[batch_size,num_node,out_channel]
        temporal_result=temporal_result.permute(0,2,1)

        return temporal_result


class ST_Conv_Block(nn.Module):
    def __init__(self,num_nodes,in_channels,out_channels,spatial_channels,kernel_size,padding=0,stride=1):
        super(ST_Conv_Block, self).__init__()

        #时序卷积模块
        self.tgc_1=Temporal_Gated_conv(in_channels,out_channels,kernel_size,padding,stride)

        #图卷积模块
        self.sgc_1 = gcn_model(num_nodes,spatial_channels,out_channels)

        #时序卷积模块
        self.tgc_2=Temporal_Gated_conv(spatial_channels,out_channels,kernel_size,padding,stride)

        #batch归一化
        self.batchNormal=nn.BatchNorm1d(out_channels)


    def forward(self,X,embding_node):
        """
        :param X: the shape of X is [batch_size,num_node,in_channel]
        :param embding_node: 图节点向量，[num_node, embding_dim]
        :return:[batch_size,nodes,out_channel]
        """

        #the shape of tgc_1_result: [batch_size,num_node,out_channel]
        tgc_1_result=self.tgc_1(X)

        # the shape of sgc_result: [batch_size,num_node,spatial_channel]
        sgc_result=self.sgc_1(tgc_1_result, embding_node)
        sgc_result=F.relu(sgc_result)

        # the shape of tgc_2_result: [batch_size,num_node,out_channel]
        tgc_2_result=self.tgc_2(sgc_result)

        # after permute, tgc_2_result:[batch_size,out_channel,nodes]
        tgc_2_result=tgc_2_result.permute(0,2,1)

        #the shape of batch_result:[batch_size,out_channel,nodes]
        batch_result=self.batchNormal(tgc_2_result)

        # after permute, batch_result:[batch_size,nodes,out_channel]
        batch_result=batch_result.permute(0,2,1)

        return batch_result





class eeg_model(nn.Module):
    def __init__(self,model_config,device):
        super(eeg_model, self).__init__()
        self.device=device
        self.embding_dim=model_config.get('embding_dim')
        self.gcn_output_dim=model_config.get('gcn_output_dim')
        self.input_dim=model_config.get('input_dim')
        self.out_channels=model_config.get('out_channels')
        self.kernel_size=model_config.get('kernel_size')
        self.padding=model_config.get('padding')
        self.stride=model_config.get('stride')
        self.output_dim=model_config.get('output_dim')
        self.num_nodes=model_config.get('num_nodes')


        #生成图节点embedding
        self.embding_node=nn.Parameter(torch.FloatTensor(self.num_nodes,self.embding_dim))

        #时空卷积模块
        self.st_block_1 = ST_Conv_Block(self.num_nodes, self.input_dim, self.out_channels, self.gcn_output_dim, self.kernel_size, self.padding,self.stride)
        self.st_block_2 = ST_Conv_Block(self.num_nodes,  self.out_channels, self.out_channels, self.gcn_output_dim, self.kernel_size, self.padding,self.stride)

        #时序卷积模块
        self.last_tgc = Temporal_Gated_conv(self.out_channels, self.out_channels, self.kernel_size, self.padding, self.stride)

        #输出层
        self.fc = nn.Linear(in_features=self.num_nodes*self.out_channels, out_features=self.output_dim)

        #初始化图节点向量
        self.init_weights(self.embding_node)


    def init_weights(self,weight):
        nn.init.xavier_normal_(weight)


    def forward(self,inputs):
        #the shape of inputs: [batch,nodes,in_feautres]
        st_result_1 = self.st_block_1(inputs, self.embding_node)
        st_result_2 = self.st_block_2(st_result_1, self.embding_node)
        out = self.last_tgc(st_result_2)
        batch,_,_=out.size()
        out=torch.reshape(out,(batch,-1))
        output=self.fc(out)
        return output
