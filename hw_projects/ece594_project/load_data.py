import torch_geometric
import torch
from pygod.utils import load_data
from pygod.generator import gen_contextual_outlier, gen_structural_outlier
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from pygod.metric import eval_roc_auc



def load_anomaly_data(data_name):
    assert(data_name in ['inj_cora','weibo','disney','books','enron'])
    return load_data(data_name)

def load_cora_dif_num(contextual_outlier, structural_outlier):
    data = Planetoid('/home/zyxue/dataset', 'Cora', transform=T.NormalizeFeatures())[0]
    data, ya = gen_contextual_outlier(data, n=contextual_outlier, k=50, seed=10)
    data, ys = gen_structural_outlier(data, n=structural_outlier, m=5)
    data.y = torch.logical_or(ys, ya).long()
    return data

