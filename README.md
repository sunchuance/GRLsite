# GATRsite
RNA–Ligand Binding Sites Prediction Using Multi-head Graph Attention Convolution Network and Pretrained RNA Language Model
# GATRsite-server
https://malab.sjtu.edu.cn/GATRsite/
# Introduction
The GATRsite consists of three parts: structure embedding module, sequence embedding module and backbone network module. We extracted node and edge features from the RNA sequence and structure information and constructed a graph representation for the RNA. As illustrated in Figure 1, in the structure embedding module, based on the three-dimensional structure of RNA, we calculate the Euclidean distance matrix and pairing matrix between bases, and then construct the adjacency matrix which can cover the spatial information between nodes. Besides, the solvent-accessible surface area of each RNA base(1D node vector) was calculated by FreeSASA to characterize the exposure degree of the base in the solvent. The sequence embedding module utilized RNAincoder(10D node vector) and RNA-MSM(768 node vector) to encode RNA sequences and generate features. In addition, the module include the 4D one-hot vector encoded nucleotide type and 2D node vector encoded molecular mass and side-chain pKa. Taken together, one 784D node vector for each residue was used to represent sequence-dependent properties. The node features were then passed through the backbone network, which was a eight-layer GATConv and the number of heads is 2. The loss computation module employed a contrastive loss function guided by binary classification loss. Finally, the classification head utilized a softmax function to transform the prediction scores of the backbone network into the node classification probabilities.
# Dependencies
torch                         1.8.1+cu111  
torch-cluster                 1.5.9  
torch-geometric               2.0.3  
torch-scatter                 2.0.6  
torch-sparse                  0.6.10  
torch-spline-conv             1.2.1  
torchaudio                    0.8.1  
torchcsprng                   0.2.0+cu111  
torchdrug                     0.2.1  
torchvision                   0.9.1+cu111
