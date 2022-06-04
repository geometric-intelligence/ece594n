import torch
from torch.nn import *
from geomstats.geometry.poincare_ball import PoincareBall

class PoincareBallModel(torch.nn.Module):
    def __init__(self, embedding_dim, num_embeddings, init_weights=1e-3, epsilon=1e-5):
        super().__init__()
        self.embedding = Embedding(num_embeddings, embedding_dim, max_norm =1., sparse=False)
        self.embedding.weight.data.uniform_(-init_weights, init_weights)
        self.epsilon = epsilon
        self.poincareBallManifold = PoincareBall(embedding_dim)

    def dist(self, u, v):
        return -1

    def forward(self, inputs):
        e = self.embeding(inputs)
        o = e.narrow(dim=1, start=1, length = e.size(1) - 1)
        #narrow-reduces the tensor 
        #expand_as
        s = e.narrow(dim=1, start=0, length=1).expand_as(o)
        
        return self.dist(s,o)