import torch
from torch.distributions import Categorical
from model import PoincareBallModel
from rsgd import RiemannianSGD
from torch.nn import CrossEntropyLoss

DIMENSIONS = 5
NEG_SAMPLES = 10
LEARNING_RATE = 0.3
LEARNING_RATE_BURNIN = LEARNING_RATE/10
NUM_BURNIN_EPOCHS = 10

# cat_dist = Categorical(probs=torch.from_numpy(weights))
unif_dist = Categorical(probs=torch.ones(len(names),) / len(names))

model = PoincareBallModel(dim=DIMENSIONS, size=len(names))
optimizer = RiemannianSGD(model.parameters())

loss_func = CrossEntropyLoss()
batch_X = torch.zeros(10, NEG_SAMPLES + 2, dtype=torch.long)
batch_y = torch.zeros(10, dtype=torch.long)

epoch = 0
while True:
    if epoch < NUM_BURNIN_EPOCHS:
        lr =LEARNING_RATE_BURNIN
        sampler = cat_dist
    else:
        lr = LEARNING_RATE
        sampler = unif_dist

    perm = torch.randperm(dataset.size(0))
    dataset_rnd = dataset[perm]
    for i in tqdm(range(0, dataset.size(0) - dataset.size(0) % 10, 10)):
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