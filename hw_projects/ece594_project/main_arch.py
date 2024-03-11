from load_detector import get_detector
from load_data import load_anomaly_data, load_cora_dif_num
import pdb
import torch_geometric
import numpy as np
import pandas as pd
from pygod.detector import OCGNN, DOMINANT
from pygod.metric import eval_roc_auc
import argparse

parser = argparse.ArgumentParser(description='Process model parameters.')
parser.add_argument('--save_path', type=str, help='the path you save the results', default='/home/zyxue/ece594_project/results/model')
args = parser.parse_args()

save_path = args.save_path

# res_array = np.zeros((len(contextual_outlier_lst), len(structural_outlier_lst)))


for method in ['Dominant','Gae','Ocgnn']:
    for data_name in ['disney','books','enron']:
        for backbone in [torch_geometric.nn.GCN,torch_geometric.nn.GAT,torch_geometric.nn.GraphSAGE]:
            file_name = f'{method}_{data_name}_{backbone.__name__}.csv'
        
            res_array = np.zeros((3, 3))
            hid_dim_lst = [64,128,256]
            num_layers_lst = [2,4,6]
            for hid_dim_idx in range(len(hid_dim_lst)):
                for num_layers_idx in range(len(num_layers_lst)): 
                    hid_dim = hid_dim_lst[hid_dim_idx]
                    num_layers = num_layers_lst[num_layers_idx]
                    data = load_anomaly_data(data_name)
                    detector = get_detector(name=method, hid_dim=hid_dim, num_layers=num_layers, epoch=200,backbone=backbone,gpu=0)
                    detector.fit(data)
                    pred, score, prob, conf = detector.predict(data,return_pred=True,
                                                    return_score=True,
                                                    return_prob=True,
                                                    return_conf=True)
                    auc_score = eval_roc_auc(data.y, score)
                    res_array[hid_dim_idx,num_layers_idx] = auc_score
                    print('AUC Score:', auc_score)
            pd.DataFrame(res_array, index=hid_dim_lst, columns=num_layers_lst).to_csv(save_path+f'/{file_name}')
            # pdb.set_trace()
# pdb.set_trace()

