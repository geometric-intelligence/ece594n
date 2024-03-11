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
parser.add_argument('--save_path', type=str, help='the path you save the results', default='/home/zyxue/ece594_project/results/data')
args = parser.parse_args()

save_path = args.save_path

contextual_outlier_lst = [5*i for i in range(1,20)]
structural_outlier_lst = [5*i for i in range(1,20)]
res_array = np.zeros((len(contextual_outlier_lst), len(structural_outlier_lst)))


for method in ['Dominant','Gae','Ocgnn']:
    for backbone in [torch_geometric.nn.GAT,torch_geometric.nn.GraphSAGE]:
        for i in range(len(contextual_outlier_lst)):
            for j in range(len(structural_outlier_lst)):
                context = contextual_outlier_lst[i]
                struct = structural_outlier_lst[j]
                data = load_cora_dif_num(context, struct)
                # print(data)
                detector = get_detector(name=method, hid_dim=64, num_layers=4, epoch=100,backbone=backbone,gpu=0)
                detector.fit(data)
                pred, score, prob, conf = detector.predict(data,return_pred=True,
                                                return_score=True,
                                                return_prob=True,
                                                return_conf=True)
                auc_score = eval_roc_auc(data.y, score)
                # print('AUC Score:', auc_score)
                res_array[i,j] = auc_score
        
        pd.DataFrame(res_array, index=contextual_outlier_lst, columns=structural_outlier_lst).to_csv(save_path+f'/{method}_{backbone.__name__}_cora_dif_num.csv')
pdb.set_trace()

