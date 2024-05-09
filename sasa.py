import os
a='/data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/data2/04-neg-9/rec'
path_list=os.listdir(a)
path_list.sort()
for ii in path_list:
    print(ii)
    os.system('freesasa /data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/data2/04-neg-9/rec/%s -n 100 --depth=residue --format=seq -o /data4_large1/home_data/ccsun/scc_neuralnetwork/06-RNA/data2/04-neg-9/sasa/%s'%(ii,ii))
