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
        sqdist = torch.sum((u - v) ** 2, dim=-1)
        squnorm = torch.sum(u ** 2, dim=-1)
        sqvnorm = torch.sum(v ** 2, dim=-1)
        x = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + self.epsilon
        z = torch.sqrt(x ** 2 - 1)
        return torch.log(x + z)

    def forward(self, inputs):
        embeds = self.embeding(inputs)
        o = embeds.narrow(dim=1, start=1, length = embeds.size(1) - 1)
        #narrow-reduces the tensor 
        #expand_as
        s = embeds.narrow(dim=1, start=0, length=1).expand_as(o)
        
        return self.dist(s,o)