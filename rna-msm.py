##########################################################################################
#                                                                                        #
#                             基于Pdb文件中没有SEQRES来进行提取rna的序列                      #
#                                                                                        #
########################################################################################## 
import os    
from Bio.PDB.PDBParser import PDBParser
# You can use a dict to convert three letter code to one letter code
d3to1 = {'A': 'A', 'U': 'U', 'C': 'C', 'G': 'G', 'LYS': 'K',}


 # Just an example input pdb

record = '/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/data2/03-complete-60/11'
path_list=os.listdir(record)
path_list.sort()
for i in path_list:
    i1=i[:4]
    print(i1)
    ii=os.path.join(record,i)
 # run parser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('struct', ii)    
    
    for model in structure:
        r_fasta=open('/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/data2/03-complete-60/%s.fasta'%i1,'w')
        #l_fasta=open('/data4_large1/home_data/ccsun/scc_neuralnetwork/02/pro-pro/zxc/pdb2/%s/fasta/%s.D.fasta'%(i1,i1),'w')
        #print(type(model))
        seq = []
        num=0
        try:
            for chain in model:
                #print(type(chain))
                for residue in chain:   
                    seq.append(d3to1[residue.resname])
                #print('>some_header\n',''.join(seq))
                #print(chain,'\n',''.join(seq),)
                a1=''.join(seq)
                i11=i1.strip()
                #print(len(i11))
                #print(a1)
                print('>',i1,'.H','\n',a1,file=r_fasta,sep='',end='')  #注意结尾的sep和end
                
                print(a1)
                print('------------------')
                seq=[]
                num+=1
        except:
            print('有配体')
        r_fasta.close() 
        
##########################################################################################
#                                                                                        #
#                                      进行MSA得到输入                                    #
#                                                                                        #
########################################################################################## 
a='/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/others-large-model/RNAcmap3/fasta2/18'
path_list=os.listdir(a)
path_list.sort()
for i in path_list[:1]:
    b=os.path.join(a,i)
    os.system('/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/others-large-model/RNAcmap3/run_rnacmap3.sh -i %s -d mfdca'%b)
    print(b)



###每个序列一个文件夹，其中包括了id.txt，results/msa比对文件，pretrained/预训练模型，最后的结果也保保存在results中
import os
a='/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/others-large-model/RNA-MSM-master/01outputs'
path_list=os.listdir(a)
path_list.sort()
for i in path_list[:1]:
    print(i)
    a1=i[:4]
    print(a1)
    a2=open('/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/others-large-model/RNA-MSM-master/01outputs/%s/rna_id.txt'%a1,'w')
    print(a1,file=a2,sep='',end='')
    a2.close()

#os.system('python3 ./others-large-model/RNA-MSM-master/RNA_MSM_Inference.py data.root_path=/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/others-large-model/RNA-MSM-master/scc data.MSA_path=/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/others-large-model/RNA-MSM-master/scc/results data.model_path=/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/others-large-model/RNA-MSM-master/scc/pretrained/RNA_MSM_pretrained.ckpt data.MSA_list=/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/others-large-model/RNA-MSM-master/scc/rna_id.txt')

#需用GPU，注意显存的问题，不能在一个机子上并行跑
import os
aa='/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/others-large-model/RNA-MSM-master/data2outputs/60'
path_list=os.listdir(aa)
path_list.sort()
for i in path_list[16:17]:
    print(i)
    os.system('python3 ./others-large-model/RNA-MSM-master/RNA_MSM_Inference.py data.root_path=/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/others-large-model/RNA-MSM-master/data2outputs/60/%s/ data.MSA_path=/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/others-large-model/RNA-MSM-master/data2outputs/60/%s/results data.model_path=/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/others-large-model/RNA-MSM-master/data2outputs/60/%s/pretrained/RNA_MSM_pretrained.ckpt data.MSA_list=/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/others-large-model/RNA-MSM-master/data2outputs/60/%s/rna_id.txt'%(i,i,i,i))


