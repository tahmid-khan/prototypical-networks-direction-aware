import torch

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
    normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
    expanded_x = normalised_x.unsqueeze(1).expand(n, m, -1)
    expanded_y = normalised_y.unsqueeze(0).expand(n, m, -1)
    cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
    cosine_dists = 1 - cosine_similarities

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    mean_sq_errors = (x - y).square().mean(dim=2)

    return mean_sq_errors + cosine_dists
