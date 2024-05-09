########################################################################################################################
#######                                                                                                     ############
#######                               生成包含特征矩阵、邻接矩阵（基于二级结构的信息）的npz文件                   ############
#######                                                                                                     ############
########################################################################################################################

import os
#from data_processing.Extract_Interface import Extract_Interface
import numpy as np
np.set_printoptions(suppress=True)  # 取消科学计数法输出
np.set_printoptions(threshold = np.inf)
import math
import operator
from functools import reduce
from scipy.spatial import distance_matrix
ee=math.e

def Prepare_Input(all_path,pos_path,neg_path,savepath):
   
    interface_rec_list=os.listdir(pos_path)
    interface_rec_list.sort()
    HH=[]
    #AA=[]
    TT=[]
    
    for num2,numm in enumerate(interface_rec_list):

        print(num2)
        interface_pdb=numm
        pdbid=interface_pdb.replace('.pdb','')
        msa='/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/others-large-model/RNA-MSM-master/data2outputs/18/%s/results/%s_emb.npy'%(pdbid,pdbid)
        rnaencoder='/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/data2/05-rnaencoder-18/%s.npy'%(pdbid)
        print(msa)
        all_pdb=os.path.join(all_path,interface_pdb)
        pos_pdb=os.path.join(pos_path,interface_pdb)
        neg_pdb=os.path.join(neg_path,interface_pdb)
        all_sasa='/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/data2/03-complete-18/sasa/%s'%interface_pdb
        pos_sasa='/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/data2/02-pos-18/sasa/%s'%interface_pdb
        neg_sasa='/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/data2/04-neg-18/sasa/%s'%interface_pdb
        print('目前正在处理的是：%s'%interface_pdb)
        
        # 归一化
        def dic_normalize(dic): #计算方法为：该值-最小值/最大值-最小值
            #print(dic)
            max_value = dic[max(dic, key=dic.get)]
            min_value = dic[min(dic, key=dic.get)]
            #print(max_value)
            #print(min_value)
            interval = float(max_value) - float(min_value)
            #print('interval',interval)
            for key in dic.keys():
                dic[key] = round((dic[key] - min_value) / interval,4)
            dic['X'] = (max_value + min_value) / 2.0
            return dic
        
        # 归一化
        def normalize_pssm(value):
            pssm=1/(1+ee**-value)
            pssm=round(pssm,4)
            return pssm
        
        #Residue type（碱基类型）
        pro_res_table = ['A', 'U', 'C', 'G']
        
        #Residue weight（分子量）
        res_weight_table = {'A': 507.18, 'U': 484.141, 'C': 483.156, 'G': 523.18}
        
        #侧链的pka（羧基的解离常数的负对数pK1）
        res_pka_table = {'A': 3.5, 'U': 9.2, 'C': 4.2, 'G': 10.8}
        
        
        res_weight_table = dic_normalize(res_weight_table)
        res_pka_table = dic_normalize(res_pka_table)
        
        
        def one_of_k_encoding(x, allowable_set):
            if x not in allowable_set:
                # print(x)
                raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
            return list(map(lambda s: x == s, allowable_set))

        
        def residue_features(residue):
            res_property2 = [res_weight_table[residue], res_pka_table[residue]]
            #res_property2 = [res_weight_table[residue]]
            features=res_property2
            return features
    
        #得到受体中每个氨基酸的SASA以及归一化后的值
        
        #print(res_list)
        #得到配体中每个氨基酸的SASA以及归一化后的值

        rec_fea=[]
        with open(all_sasa,'r') as rec_file:
            line = rec_file.readlines() 
            for m,i in enumerate(line[1:-1]):
                i=i.strip()
                resid=i[14:15]
                #print(resid)
                rec_feature=residue_features(resid)
                
                pro_seq=resid
                pro_hot = np.zeros((1, len(pro_res_table)))
                for i in range(len(pro_seq)):
                    pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
                pro_hot1=reduce(operator.add, pro_hot)  #将二维将至一维
                pro_hot1=pro_hot1.tolist()
                rec_fea.append(pro_hot1+rec_feature)
            rec_fea=np.array(rec_fea) 
            shape1=np.shape(rec_fea)
            print('最初特征矩阵的大小为：',np.shape(rec_fea))
            #print('得到受体的特征矩阵：','\n',rec_fea)
        
        
        msa1=np.load(msa)
        #msa2=msa1[:-1]  #因为前期生成序列的时候在最后都多一个碱基，所以在这里删除它
        merge_rec_fea=np.concatenate((rec_fea, msa1), axis=1)
        print('第一次合并后的特征矩阵大小为：',np.shape(merge_rec_fea))
        #print(merge_rec_fea)
        rnaencoder1=np.load(rnaencoder)
        rnaencoder2= rnaencoder1[0, ::]
        length=shape1[0]
        rnaencoder3=rnaencoder2[:length,]
        merge_rec_fea2=np.concatenate((merge_rec_fea, rnaencoder3), axis=1)
        print('再次合并后的特征矩阵大小为：',np.shape(merge_rec_fea2))
        
        ###把AG的N1和CU的N3的坐标，原子类型，碱基类型取出来
        c1_accord_rec=[]
        all_basic1=[]
        aaa=[]
        with open(all_pdb,'r') as file:
            file=file.readlines()
            for i in file:
                x=[float(i[31:38])]
                y=[float(i[39:46])]
                z=[float(i[47:54])]
                atom=i[13:15]
                basic_type=i[19:20]
                residue_id = int(i[23:26])
                #print(atom)
                if basic_type=='A' and atom=='N1':
                    c1_accord_rec.append(x+y+z)
                    all_basic1.append(basic_type)
                    aaa.append(residue_id)
                if basic_type=='G' and atom=='N1':
                    c1_accord_rec.append(x+y+z)
                    all_basic1.append(basic_type)
                    aaa.append(residue_id)
                if basic_type=='C' and atom=='N3':
                    c1_accord_rec.append(x+y+z)
                    all_basic1.append(basic_type)
                    aaa.append(residue_id)
                if basic_type=='U' and atom=='N3':
                    c1_accord_rec.append(x+y+z)
                    all_basic1.append(basic_type)
                    aaa.append(residue_id)
            #print(c1_accord_rec)
            #print(all_basic1)
            c1_accord_rec=np.array(c1_accord_rec)
        print(aaa)    
        all_basic2=all_basic1
        
        ligand_count=len(c1_accord_rec)
        #基于坐标计算距离
        dis_rec_rec = distance_matrix(c1_accord_rec, c1_accord_rec)
        print(np.shape(dis_rec_rec))
        
        num1=np.where(dis_rec_rec<=3.2)  ##输出小于3.2的索引，x1是横坐标，y1是纵坐标
        x1=num1[0]
        y1=num1[1]
        #print(x1)
        #print(y1)
        #print(len(x1))
        for i in range(len(x1)): ##用碱基配对作为条件，小于3.2的为1，否则有的不是配对的碱基，距离也小于3.2，还有一种情况，相邻的两个碱基，如CG，其距离一般会比3.2要大，所以这里选择3.2
            #print(x1[i],y1[i])
            all_basic11=all_basic1[x1[i]]
            all_basic22=all_basic2[y1[i]]
            total1=all_basic11+all_basic22
            #print(total1)
            #dis_rec_rec[x1[i]][y1[i]]=1
            
            if total1=='AU' or total1=='UA' or total1=='CG' or total1=='GC':
                #print(total1)
                dis_rec_rec[x1[i]][y1[i]]=1
        #print(dis_rec_rec)
        
        
        for ii in np.nditer(dis_rec_rec,op_flags=['readwrite']):
           
            if ii >1.1:
                #print(i1)
                ii[...]=0
        #print(dis_rec_rec)
        print('邻接矩阵的大小为：',np.shape(dis_rec_rec))
        
        #print(dis_rec_rec)
        #处理每个节点的标签，利用sasa文件中的resid，pos，neg，all这三个sasa文件
        all_target=[]
        with open(all_sasa,'r') as tar_file:
            line=tar_file.readlines()
            for i in line[1:-1]:
                i=i.strip() 
                all_id=i[6:10]
                all_id=int(all_id)
                all_target.append(all_id)
        #print(all_target)
        
        pos_target=[]
        with open(pos_sasa,'r') as tar_file1:
            line1=tar_file1.readlines()
            for i1 in line1[1:-1]:
                i1=i1.strip() 
                pos_id=i1[6:10]
                pos_id=int(pos_id)
                pos_target.append(pos_id)
        #print(pos_target)
        
        neg_target=[]
        with open(neg_sasa,'r') as tar_file2:
            line2=tar_file2.readlines()
            for i2 in line2[1:-1]:
                i2=i2.strip() 
                neg_id=i2[6:10]
                neg_id=int(neg_id)
                neg_target.append(neg_id)
        
        tt=all_target
        new_target=all_target
        print(new_target)
        for numm,i55 in enumerate(tt):
            for i66 in neg_target:
                if i66==i55:
                    new_target[numm]=0
        for num,i5 in enumerate(tt):
            for i6 in pos_target:
                if i6==i5:
                    new_target[num]=1
        #print(new_target)
        
        
        for i in merge_rec_fea2:
            i=i.tolist() 
            #print(type(i))
            HH.append(i)
            
            
        if num2==0:
            print('走这')
            A=dis_rec_rec
            #print(A)
        if num2>0:
            print('zouzheli')
            B=dis_rec_rec
            #print('这是A',A)
            #print('这是B',B)
            size_a=np.shape(A)
            size_b=np.shape(B)
            #print(size_a)
            #print(size_b)
            len_a=size_a[0]
            len_b=size_b[0]
            len_new=len_a+len_b
            m1=np.zeros((len_new,len_new))
            #print(m1)
            m1[:size_a[0],:size_a[0]]=A
            #print(m1)
            #print(m1[2:,2:])
            m1[size_a[0]:,size_a[0]:]=B
            #print(m1)
            print(np.shape(m1))
            A=m1
        
        for i in new_target:
            TT.append(i)
        print(TT)   
        print('=============================================')
        
    HH=np.array(HH) #特征矩阵逐渐增加
    #AA=np.array(AA)
    print('总的特征矩阵的大小为：',np.shape(HH))
    print('总的邻接矩阵的大小为',np.shape(A))
    print('总的标签矩阵的大小为',np.shape(TT))
    #print(TT)
    pdbidd='test'
    input_file=os.path.join(savepath,"%s.npz"%pdbidd)
    np.savez(input_file,  H=HH, A1=A, T=TT)
    
Prepare_Input(all_path='/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/data2/03-complete-18/rec',
             pos_path='/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/data2/02-pos-18/rec',
             neg_path='/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/data2/04-neg-18/rec',
             savepath='/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/data2/07-npz-18-pair/784/test-npz')


