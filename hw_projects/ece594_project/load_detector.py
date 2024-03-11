import torch_geometric
from pygod.detector import DOMINANT, OCGNN, GAE, AnomalyDAE, GADNR, DMGD, CoLA



def get_detector(name, hid_dim=64, num_layers=4, epoch=100,backbone=torch_geometric.nn.GCN,gpu=0):
    if name == 'Dominant':
        return DOMINANT(hid_dim=hid_dim, num_layers=num_layers, epoch=epoch, backbone=backbone, gpu=gpu)
    elif name == 'Ocgnn':
        return OCGNN(hid_dim=hid_dim, num_layers=num_layers, epoch=epoch, backbone=backbone, gpu=gpu)
    elif name == 'Gae':
        return GAE(hid_dim=hid_dim, num_layers=num_layers, epoch=epoch, backbone=backbone, gpu=gpu)
    elif name == 'CoLA':
        return CoLA(hid_dim=hid_dim, num_layers=num_layers, epoch=epoch, backbone=backbone, gpu=gpu)
    elif name == 'AnomalyDAE':
        return AnomalyDAE(gpu=gpu)
    elif name == 'DMGD':
        return DMGD(hid_dim=hid_dim, num_layers=num_layers, epoch=epoch, backbone=backbone, gpu=-1)
    else:
        raise ValueError(f'Unknown detector: {name}')



