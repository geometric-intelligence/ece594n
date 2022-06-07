# https://github.com/facebookresearch/poincare-embeddings

# https://github.com/geomstats/geomstats/blob/a8f1d947608f1e116509f74da69b2f012de83e21/examples/learning_graph_embedding_and_predicting.py
# https://github.com/geomstats/geomstats/blob/112dd8a29cd5712cbbd3627c976a9a4d77bd629d/geomstats/geometry/poincare_ball.py

#https://lars76.github.io/2020/07/24/implementing-poincare-embedding.html
# https://lars76.github.io/2020/07/23/rsgd-in-pytorch.html

#Gensim poincare embeddings
# https://radimrehurek.com/gensim/models/poincare.html
import torch
from torch.distributions import Categorical
from rsgd import RiemannianSGD
from torch.nn import CrossEntropyLoss, Embedding
import tqdm
import pandas as pd
import os

DIMENSIONS = 2
NEG_SAMPLES = 10
# def dist(self, u, v):


# def main():
#     print("MAIN")

# def loss(output, target):
#     loss = 3
#     # loss = torch.sum(torch.log()
#     return loss

# def soft_ranking_loss(output, target):

def main():
    epoch = 0
    neighbors = []
    dataset, names = getDataset()
    # cat_dist = Categorical(probs=torch.from_numpy(weights))
    # unif_dist = Categorical(probs=torch.ones(len(names),) / len(names))
    unif_dist = Categorical(probs=torch.ones(names.shape[0],) / names.shape[0])

    model = Model(dim=DIMENSIONS, size=names.shape[0])
    optimizer = RiemannianSGD(model.parameters())

    loss_func = CrossEntropyLoss()
    batch_X = torch.zeros(10, NEG_SAMPLES + 2, dtype=torch.long)
    batch_y = torch.zeros(10, dtype=torch.long)

    while True:
        if epoch < 20:
            lr = 0.003
            # sampler = cat_dist
            sampler = unif_dist
        elif epoch < 30:
            lr = 0.3
            sampler = unif_dist
        else: 
            print("done\n")
            return
        print(dataset.shape[0])
        perm = torch.randperm(dataset.shape[0])
        print(perm)
        dataset_rnd = dataset[perm]
        for i in tqdm(range(0, dataset.shape[0] - dataset.shape[0] % 10, 10)):
            batch_X[:,:2] = dataset_rnd[i : i + 10]

            for j in range(10):
                a = set(sampler.sample([2 * NEG_SAMPLES]).numpy())
                negatives = list(a - (set(neighbors[batch_X[j, 0]]) | set(neighbors[batch_X[j, 1]])))
                batch_X[j, 2 : len(negatives)+2] = torch.LongTensor(negatives[:NEG_SAMPLES])

            optimizer.zero_grad()
            preds = model(batch_X)

            loss = loss_func(preds.neg(), batch_y)
            loss.backward()
            optimizer.step(lr=lr)
        epoch += 1 

def getDataset():
    results = pd.read_csv(os.getcwd()+"/wordnet/mammal_closure_noweights.csv")
    npArray = results.to_numpy()
    print(npArray)
    names = results.stack()
    names = names.unique()
    print(names.shape)
    return torch.from_numpy(npArray), names


class Model(torch.nn.Module):
    def __init__(self, dim, size, init_weights=1e-3, epsilon=1e-7):
        super().__init__()
        self.embedding = Embedding(size, dim, sparse=False)
        self.embedding.weight.data.uniform_(-init_weights, init_weights)
        self.epsilon = epsilon

    def dist(self, u, v):
        sqdist = torch.sum((u - v) ** 2, dim=-1)
        squnorm = torch.sum(u ** 2, dim=-1)
        sqvnorm = torch.sum(v ** 2, dim=-1)
        x = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + self.epsilon
        z = torch.sqrt(x ** 2 - 1)
        return torch.log(x + z)

    def forward(self, inputs):
        e = self.embedding(inputs)
        o = e.narrow(dim=1, start=1, length=e.size(1) - 1)
        s = e.narrow(dim=1, start=0, length=1).expand_as(o)

        return self.dist(s, o)



if __name__ == "__main__":
    main();

    