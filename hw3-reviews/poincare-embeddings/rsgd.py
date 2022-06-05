import torch
from torch.optim import Optimizer

@torch.jit.script
def lambda_x(x: torch.Tensor):
    return 2 / (1 - torch.sum(x ** 2, dim=-1, keepdim=True))

@torch.jit.script
def mobius_add(x: torch.Tensor, y: torch.Tensor):
    x2 = torch.sum(x ** 2, dim=-1, keepdim=True)
    y2 = torch.sum(y ** 2, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)

    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    denom = 1 + 2 * xy + x2 * y2

    return num / denom.clamp_min(1e-15)

@torch.jit.script
def expm(p: torch.Tensor, u: torch.Tensor):
    return p + u
    # for exact exponential mapping
    #norm = torch.sqrt(torch.sum(u ** 2, dim=-1, keepdim=True))
    #return mobius_add(p, torch.tanh(0.5 * lambda_x(p) * norm) * u / norm.clamp_min(1e-15))

@torch.jit.script
def grad(p: torch.Tensor):
    p_sqnorm = torch.sum(p.data ** 2, dim=-1, keepdim=True)
    return p.grad.data * ((1 - p_sqnorm) ** 2 / 4).expand_as(p.grad.data)

class RiemannianSGD(Optimizer):
    def __init__(self, params):
        super(RiemannianSGD, self).__init__(params, {})

    def step(self, lr=0.3):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = grad(p)
                d_p.mul_(-lr)

                p.data.copy_(expm(p.data, d_p))