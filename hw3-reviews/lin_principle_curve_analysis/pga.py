# Code Adapted from Nicolas Guigui's Work
import math
import os
import torch.optim as optim
import matplotlib.pyplot as plt
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean


def model(x, tangent_vec, base_point, metric):
    times = x[:, None] if metric.default_point_type == 'vector' else \
        x[:, None, None]
    return metric.exp(times * tangent_vec[None], base_point)


def projection(point, tangent_vec, base_point, metric, n_samples, max_iter=100, tol=1e-6):

    def loss(param):
        projected = model(param, tangent_vec, base_point, metric)
        return gs.sum(metric.squared_dist(point, projected))

    # value_and_grad = gs.autograd.value_and_grad(loss)
    parameter = 0 if point.ndim == 1 else gs.zeros(len(point))
    parameter = parameter.requires_grad_(True)

    opt = optim.Adam([parameter], lr=0.1)

    e = 0
    loss_at_param, criterion = math.inf, math.inf
    for e in range(max_iter):
        loss_at_param = loss(parameter)
        opt.zero_grad()
        loss_at_param.backward(retain_graph=True)
        criterion = parameter.grad.detach().norm() / n_samples
        if criterion <= tol:
            print('Projection: Convergence tol reached ')
            break
        opt.step()
        if gs.any(gs.isnan(parameter)):
            print('Nan')
    if e == max_iter - 1:
        print(f'Max iter reached with gradient norm: {criterion}')
    print('Final loss:', loss_at_param.detach())
    return model(parameter, tangent_vec, base_point, metric), parameter.detach()


# proj, coefs = projection(target, beta_hat, intercept_hat, max_iter=1000)


def pga(point, base_point, space, metric, n_samples, max_iter=100, tol=1e-10):

    def loss(param):
        tangent_vec = space.to_tangent(param, base_point)
        projected_, _ = projection(point, tangent_vec, base_point, metric, n_samples = n_samples)
        return gs.sum(metric.squared_dist(point, projected_)) # + (
                # gs.sum((param - tangent_vec) ** 2)) + gs.maximum(
                # gs.sum(param ** 2) - gs.pi ** 2, 0)

    # value_and_grad = gs.autograd.value_and_grad(loss)
    parameter = 0 if point.ndim == 1 else gs.random.rand(point[-1].shape[0])
    parameter = parameter.requires_grad_(True)

    opt = optim.Adam([parameter], lr=1)

    e = 0
    errors = 0
    loss_at_param, criterion = math.inf, math.inf
    previous_loss = loss_at_param
    previous_parameter = parameter.detach().clone()
    for e in range(max_iter):
        loss_at_param = loss(parameter)
        print('pga_loss', loss_at_param)
        if loss_at_param > previous_loss:
            errors += 1
            if errors == 3:
                print('breaking')
                break

        if gs.any(gs.isnan(loss_at_param)):
            print('NaN')
            break
        opt.zero_grad()
        loss_at_param.backward()
        criterion = parameter.grad.detach().norm() / n_samples
        if criterion <= tol:
            print('PGA: Convergence tol reached')
            break
        previous_parameter = parameter.detach().clone()
        previous_loss = loss_at_param
        opt.step()

    if e == max_iter - 1:
        print('Max iter reached in PGA with grad', criterion)
    print('Final loss:', loss_at_param)
    tangent_vec_final = space.to_tangent(previous_parameter, base_point)
    projected, times = projection(point, tangent_vec_final, base_point, metric , n_samples= n_samples)
    return projected.detach(), times.detach(), previous_parameter


