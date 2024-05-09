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
np.set_printoptions(threshold=np.inf)
from torch_geometric.data import Data
from torch.nn import Linear
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,SAGEConv
import os.path as osp
from torch_geometric.nn import global_mean_pool
import scipy.sparse as sp
import torch_geometric.nn as pyg_nn
from torch_geometric.data import DataLoader
import warnings
warnings.filterwarnings("ignore", category=Warning)
from sklearn.metrics import roc_auc_score
np.set_printoptions(suppress=True)
#from model import Net  #注意Net的导入，是在当时训练的那个脚本里面


#sccmodel=torch.load('/data2/data_home/ccsun/link/scc_neuralnetwork/Recognition_of_antigens_by_antibodies/try1/npz_processed_grb/cdr_adj/savemodel/t2/99_model.pt')

#sccmodel=torch.load('/data4_large1/home_data/ccsun/scc_neuralnetwork/02/train/grb/savemodel-normal2-desk/58_model.pt')
#sccmodel.eval()
###注意：如果在GPU上训练的模型则需要在加载模型的时候，加上map_location='cpu'
class Net(torch.nn.Module):
    """构造GCN模型网络"""
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = SAGEConv(784, 64) # 构造第一层，输入和输出通道，输入通道的大小和节点的特征维度一致
        #self.conv2 = SAGEConv(64, 128) 
        #self.conv3 = SAGEConv(128, 2) 
        self.conv1 = GATConv(784, 4, heads=2, concat=False) # 构造第一层，输入和输出通道，输入通道的大小和节点的特征维度一致
        self.conv2 = GATConv(4, 16, heads=2, concat=False) 
        self.conv3 = GATConv(16, 32, heads=2, concat=False) 
        self.conv4 = GATConv(32, 64, heads=2, concat=False)
        self.conv5 = GATConv(64, 128, heads=2, concat=False)
        self.conv6 = GATConv(128, 256, heads=2, concat=False)
        self.conv7 = GATConv(256, 512, heads=2, concat=False)
        self.conv8 = GATConv(512, 2, heads=2, concat=False) 

        #self.conv4 = SAGEConv(128, 256)
        #self.conv5 = SAGEConv(256, 512)

        #self.conv6 = SAGEConv(512,2)
        # 构造第三层，输入和输出通道，输出通道的大小和图或者节点的分类数量一致，比如此程序中图标记就是二分类0和1，所以等于2
    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        #x = self.conv2(x, edge_index)
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        #x = self.conv3(x, edge_index)
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        #x = self.conv3(x, edge_index)
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv4(x, edge_index))
        #x = self.conv3(x, edge_index)

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv5(x, edge_index))
        #x = self.conv3(x, edge_index)

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv6(x, edge_index))
        #x = self.conv7(x, edge_index)
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv7(x, edge_index))
        x = self.conv8(x, edge_index)
        #x = self.conv6(x, edge_index)
        #x = x.relu()
        #x = F.dropout(x, training=self.training)
        #print(np.shape(x))
        #x = pyg_nn.global_mean_pool(x, batch) # 池化降维，根据batch的值知道有多少张图片（每张图片的节点的分类值不同0-19），再将每张图片的节点取一个全局最大的节点作为该张图片的一个输出值
        #print(np.shape(x))
        #return x
        return F.softmax(x, dim=1) # softmax可以得到每张图片的概率分布，设置dim=1，可以看到每一行的加和为1，再取对数矩阵每个结果的值域变成负无穷到0

test_loader = DataLoader(dataset1, batch_size=1, shuffle=False)
preds=[]
prob_all=[]
label_all=[]
device = torch.device('cuda:2') #用GPU
def evaluate(loader): # 构造测试函数
    sccmodel=Net()
    #PATH='/data4_large1/home_data/ccsun/scc_neuralnetwork/02/train/grb/savemodel-normal2-desk/25_model.pt'
    #sccmodel=torch.load('/data4_large1/home_data/ccsun/scc_neuralnetwork/GNN-ranking-decoys/train/gab/savemodel-normal/20_model.pt',map_location='cpu')
    #sccmodel=torch.load('/data4_large1/home_data/ccsun/scc_neuralnetwork/GNN-ranking-decoys/atom-based/save_model/t1-graphSAGE/30_model.pt',map_location='cpu') #old
    
    #sccmodel=torch.load('/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/data2/010-log/best-8layer-6/552_model.pt')  
    sccmodel=torch.load('/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/data2/010-log/best-8layer-3/2164_model.pt') 
    #sccmodel=torch.load('/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/py/785-1/4559_model.pt') 
    
    sccmodel.eval()
    #print(sccmodel)   
    correct = 0
    with torch.no_grad():
        
        a1=[]
        a2=[]
        a3=[]
        a4=[]
        correct = 0
        for data in loader:
            # Iterate in batches over the training/test dataset.
    
            data=data.to(device)
    
            out = sccmodel(data.x, data.edge_index, data.batch)
            #print(out)
            pred = out.argmax(dim=1).detach().cpu()  # 返回最大值对应的索引值
            print(pred)
            a1.extend(pred)
        
            #aa=data.y.detach().cpu()
            #a2.extend(aa)
        #print(a1)
        for i in a1:
            i=int(i)
           
            a3.append(i)
           
        #print(a3)
        #print(a4)
        #return precision_score(a4,a3), recall_score(a4,a3),matthews_corrcoef(a4,a3)


testacc=evaluate(test_loader)
print('----------------------')
print(testacc)
