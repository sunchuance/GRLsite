import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import download_url
import os
from torch_geometric.io import read_planetoid_data
from torch_geometric.datasets import Planetoid
import numpy as np
from torch_geometric.data import Data
from torch.nn import Linear
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,SAGEConv,GATConv
import os.path as osp
from torch_geometric.nn import global_mean_pool
import scipy.sparse as sp
import torch_geometric.nn as pyg_nn
from torch_geometric.data import DataLoader
import warnings
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import matthews_corrcoef

warnings.filterwarnings("ignore", category=Warning)
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import matplotlib
from matplotlib.pyplot import MultipleLocator

file='/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/data2/07-npz-60-pair/784/train/'

class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        

    # 返回数据集源文件名，告诉原始的数据集存放在哪个文件夹下面，如果数据集已经存放进去了，那么就会直接从raw文件夹中读取。
    @property
    def raw_file_names(self):
        # pass # 不能使用pass，会报join() argument must be str or bytes, not 'NoneType'错误
        return []

    # 首先寻找processed_paths[0]路径下的文件名也就是之前process方法保存的文件名
    @property
    def processed_file_names(self):
        return ['datas0.pt','datas1.pt']

    # 用于从网上下载数据集，下载原始数据到指定的文件夹下，自己的数据集可以跳过
    def download(self):
        pass
    def process(self):
        #
        data_list=[]
        for i in range(1):
            path_list=os.listdir(file)
            path_list.sort()
            npz=path_list[i]
            #print(npz)
            #pdb2=path_list[i]
            aa=os.path.join(file,npz)
    
            fileload=np.load(aa)  #读取npz文件
            key=list(fileload.keys()) #获取npz里面的键，由一个个字典组成
            #print(key)

            node=fileload['H'] #获取节点以及节点特征，数据类型为numpy.ndarray
            #node = torch.tensor(x) #数据类型为torch.Tensor
            #print(np.shape(node))

            adj=fileload['A1'] #data.edge_index节点和边的邻接矩阵
            #adj2=fileload['A2'] #data.edge_attr: 边属性
    

            #邻接矩阵转换成COO稀疏矩阵及转换  
            edge_index_temp = sp.coo_matrix(adj)  
            #print('edge_index_temp为：')
            #print(np.shape(edge_index_temp))
            tar=fileload['T']
    
    
            indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
            edge_index = torch.LongTensor(indices)
            #print(np.shape(edge_index))
            #节点及节点特征数据转换
            x = node
            #print(x)
            #x = x.squeeze(0)
            x = torch.FloatTensor(x)
            #print(np.shape(x))
            a=tar
            y=torch.LongTensor(a)
    
            #edge_attr=adj2

            #构建数据集:为一张图，节点数量，节点特征，Coo稀疏矩阵的边(邻接矩阵)，边的特征矩阵,一个图一个标签
            data=Data(x=x, edge_index=edge_index,y=y)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            torch.save(data,osp.join(self.processed_dir,'datas{}.pt'.format(i)))
    
    def len(self):
        return 1

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'datas{}.pt'.format(idx)))
        return data


file1='/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/data2/07-npz-9-pair/784/test-npz'

class MyOwnDataset1(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        

    # 返回数据集源文件名，告诉原始的数据集存放在哪个文件夹下面，如果数据集已经存放进去了，那么就会直接从raw文件夹中读取。
    @property
    def raw_file_names(self):
        # pass # 不能使用pass，会报join() argument must be str or bytes, not 'NoneType'错误
        return []

    # 首先寻找processed_paths[0]路径下的文件名也就是之前process方法保存的文件名
    @property
    def processed_file_names(self):
        return ['datas0.pt','datas1.pt']

    # 用于从网上下载数据集，下载原始数据到指定的文件夹下，自己的数据集可以跳过
    def download(self):
        pass
    def process(self):
        #
        data_list=[]
        for i in range(1):
            path_list=os.listdir(file1)
            path_list.sort()
            npz=path_list[i]
            #print(npz)
            #pdb2=path_list[i]
            aa=os.path.join(file1,npz)
    
            fileload=np.load(aa)  #读取npz文件
            key=list(fileload.keys()) #获取npz里面的键，由一个个字典组成
            #print(key)

            node=fileload['H'] #获取节点以及节点特征，数据类型为numpy.ndarray
            #node = torch.tensor(x) #数据类型为torch.Tensor
            #print(np.shape(node))

            adj=fileload['A1'] #data.edge_index节点和边的邻接矩阵
            #adj2=fileload['A2'] #data.edge_attr: 边属性
    

            #邻接矩阵转换成COO稀疏矩阵及转换  
            edge_index_temp = sp.coo_matrix(adj)  
            #print('edge_index_temp为：')
            #print(np.shape(edge_index_temp))
            tar=fileload['T']
    
    
            indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
            edge_index = torch.LongTensor(indices)
            #print(np.shape(edge_index))
            #节点及节点特征数据转换
            x = node
            #print(x)
            #x = x.squeeze(0)
            x = torch.FloatTensor(x)
            #print(np.shape(x))
            a=tar
            y=torch.LongTensor(a)
    
            #edge_attr=adj2

            #构建数据集:为一张图，节点数量，节点特征，Coo稀疏矩阵的边(邻接矩阵)，边的特征矩阵,一个图一个标签
            data=Data(x=x, edge_index=edge_index,y=y)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            torch.save(data,osp.join(self.processed_dir,'datas{}.pt'.format(i)))
    
    def len(self):
        return 1

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'datas{}.pt'.format(idx)))
        return data

dataset=MyOwnDataset('/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/data2/07-npz-60-pair/784/train-pt')
dataset1=MyOwnDataset1('/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/data2/07-npz-9-pair/784/test-pt')
class Net(torch.nn.Module):
    """构造GCN模型网络"""
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = GCNConv(784, 128) # 构造第一层，输入和输出通道，输入通道的大小和节点的特征维度一致
        #self.conv2 = GCNConv(128, 256) 
        #self.conv3 = GCNConv(256, 2) 
        self.conv1 = GATConv(784, 128, heads=1, concat=False) # 构造第一层，输入和输出通道，输入通道的大小和节点的特征维度一致
        self.conv2 = GATConv(128, 256, heads=1, concat=False) 
        self.conv3 = GATConv(256, 2, heads=1, concat=False) 
        #self.conv4 = GATConv(256, 2, heads=2, concat=False) 

        #self.conv4 = SAGEConv(128, 256)
        #self.conv5 = SAGEConv(256, 512)

        #self.conv6 = SAGEConv(512,2)
        # 构造第三层，输入和输出通道，输出通道的大小和图或者节点的分类数量一致，比如此程序中图标记就是二分类0和1，所以等于2
    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        #x = self.conv2(x, edge_index)
        
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        
        
        #x = self.conv6(x, edge_index)
        #x = x.relu()
        #x = F.dropout(x, training=self.training)
        #print(np.shape(x))
        #x = pyg_nn.global_mean_pool(x, batch) # 池化降维，根据batch的值知道有多少张图片（每张图片的节点的分类值不同0-19），再将每张图片的节点取一个全局最大的节点作为该张图片的一个输出值
        #print(np.shape(x))
        #return x
        return F.softmax(x, dim=1) # softmax可以得到每张图片的概率分布，设置dim=1，可以看到每一行的加和为1，再取对数矩阵每个结果的值域变成负无穷到0


# 构建模型实例
model = Net() # 构建模型实例

device = torch.device('cuda:2') #用GPU
model = Net().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # 优化器，参数优化计算
train_loader = DataLoader(dataset, batch_size=64,shuffle=True) # 加载训练数据集，训练数据中分成每批次n个图片data数据
for data in train_loader:
    print(data)
test_loader = DataLoader(dataset1, batch_size=64, shuffle=False)
criterion = torch.nn.CrossEntropyLoss()
criterion = criterion.to(device)

def train():
    model.train() # 表示模型开始训练
    loss_all = 0
    # 一轮epoch优化的内容
    for data in train_loader: # 每次提取训练数据集一批20张data图片数据赋值给data
        # data是batch_size图片的大小
        #print(data.x)

        data=data.to(device)

        output = model(data.x, data.edge_index, data.batch) # 前向传播，把一批训练数据集导入模型并返回输出结果，输出结果的维度是[20,2]
        label = data.y # 20张图片数据的标签集合，维度是[20]
        #print(label)
        #print(output)
        loss = criterion(output,label) # 损失函数计算，
        loss.backward() #反向传播
        loss_all += loss.item() # 将最后的损失值汇总
        optimizer.step() # 更新模型参数
        optimizer.zero_grad() # 梯度清零
    train_loss = (loss_all / len(dataset)) # 算出损失值或者错误率
 
    return train_loss
    
#模型的保存    
#torch.save(model, "E:\GCNmodel\model\MyGCNmodel.pt")
        
# 测试
def evaluate(loader): # 构造测试函数计算acc
    model.eval()
    preds=[]
    tru=[]
    correct = 0
    for data in loader:
        # Iterate in batches over the training/test dataset.

        data=data.to(device)

        out = model(data.x, data.edge_index, data.batch)
        #print(out)
        pred = out.argmax(dim=1)  # 返回最大值对应的索引值
       # print('预测的是:',pred)
        aa=data.y
        for i,ii in zip(pred,aa):
            i=int(i)
            ii=int(ii)
            preds.append(ii)
            if i==ii:
                tru.append(ii)
       # print('实际的是：',aa)
        correct=len(tru)
        all1=len(preds)
        
    return correct / all1  # Derive ratio of correct predictions.
def evaluatee(loader): # 构造测试函数计算acc
    model.eval()
    preds=[]
    tru=[]
    correct = 0
    for data in loader:
        # Iterate in batches over the training/test dataset.

        data=data.to(device)

        out = model(data.x, data.edge_index, data.batch)
        #print(out)
        pred = out.argmax(dim=1)  # 返回最大值对应的索引值
        #print('预测的是:',pred)
        aa=data.y
        for i,ii in zip(pred,aa):
            i=int(i)
            ii=int(ii)
            preds.append(ii)
            if i==ii:
                tru.append(ii)
        #print('实际的是：',aa)
        correct=len(tru)
        all1=len(preds)
        
    return correct / all1  # Derive ratio of correct predictions.

def evaluate1(loader): # 构造测试函数计算auc
    model.eval()
    prob_all=[]
    label_all=[]
    correct = 0
    for data in loader:
        # Iterate in batches over the training/test dataset.

        data=data.to(device)

        out = model(data.x, data.edge_index, data.batch)
        #print(out)
        aa=data.y.detach().cpu()
        #print(aa)
        prob_all.extend(out[:,1].detach().cpu().numpy()) #分数必须是具有较大标签的类的分数，通俗点理解:模型打分的第二列
        label_all.extend(aa)
        
    return roc_auc_score(label_all,prob_all)# Derive ratio of correct pre

def evaluate11(loader): # 构造测试函数计算auc
    model.eval()
    prob_all=[]
    label_all=[]
    correct = 0
    for data in loader:
        # Iterate in batches over the training/test dataset.

        data=data.to(device)

        out = model(data.x, data.edge_index, data.batch)
        #print(out)
        aa=data.y.detach().cpu()
        #print(aa)
        prob_all.extend(out[:,1].detach().cpu().numpy()) #分数必须是具有较大标签的类的分数，通俗点理解:模型打分的第二列
        label_all.extend(aa)
        
    return roc_auc_score(label_all,prob_all)


def evaluate2(loader): # 构造测试函数计算精确率,召回率,f1_score
    model.eval()
    a1=[]
    a2=[]
    a3=[]
    a4=[]
    correct = 0
    for data in loader:
        # Iterate in batches over the training/test dataset.

        data=data.to(device)

        out = model(data.x, data.edge_index, data.batch)
        #print(out)
        pred = out.argmax(dim=1).detach().cpu()  # 返回最大值对应的索引值
        #print(pred)
        a1.extend(pred)
    
        aa=data.y.detach().cpu()
        a2.extend(aa)
    #print(a2)
    for i,ii in zip(a1,a2):
        i=int(i)
        ii=int(ii)
        #print(type(i))
        a3.append(i)
        a4.append(ii)
    #print(a3)
    #print(a4)
    return precision_score(a4,a3), recall_score(a4,a3),matthews_corrcoef(a4,a3)

#开始训练
train_loss_all = []
#valid_loss_all = []

train_acc_all=[]
train_auc_all=[]

#valid_acc_all=[]
test_acc_all=[]
test_auc_all=[]

test_recision_all=[]
test_recall_all=[]
test_mcc_all=[]
for epoch in range(1):
    train_loss=train()	
    train_acc = evaluate(train_loader)
    train_auc = evaluate1(train_loader)
    #a3=evaluate2(train_loader)
    #print('a3',a3)
    #train_recision = a3[0]
   # train_recall = a3[1]
    #print('===================================================================================')
    test_acc = evaluatee(test_loader)
    test_auc = evaluate11(test_loader)
    
    a4=evaluate2(test_loader)
    #print(a4)
    test_recision = a4[0]
    test_recall = a4[1]
    test_mcc = a4[2]
    
    train_loss_all.append(train_loss)
    train_acc_all.append(train_acc)
    train_auc_all.append(train_auc)
    test_acc_all.append(test_acc)
    test_auc_all.append(test_auc)
    test_recision_all.append(test_recision)
    test_recall_all.append(test_recall)
    test_mcc_all.append(test_mcc)
    
    #if test_acc >= 0.7:
    #    print('测试集的准确率为%s，在第%d中'%(t_acc,epoch))
    torch.save(model, "/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/data2/010-log/t2-save/%d_model.pt"%epoch)
    print(f'Epoch: {epoch:03d}, train loss: {train_loss:.4f},train Acc: {train_acc:.4f},train AUC: {train_auc:.4f},test Acc: {test_acc:.4f},test AUC: {test_auc:.4f},test prescision:{test_recision:.4f} ,test recall:{test_recall:.4f},test mcc:{test_mcc:.4f}')
    print('===========================================')



