
import numpy as np      

def gaussian_kernel(shape, sigma, center=None):
    h, w = shape
    if center is None:
        center = ((h-1)/2, (w-1)/2)

    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    cy, cx = center
    dist2 = (y - cy)**2 + (x - cx)**2
    return np.exp(-dist2 / (2 * sigma**2))

def negative_gaussian(grid_size, sigma, center=None):
    h, w = grid_size
    if center is None:
        center = ((h - 1) / 2, (w - 1) / 2)

    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    cy, cx = center

    dist2 = (y - cy)**2 + (x - cx)**2
    g = np.exp(-dist2 / (2 * sigma**2))

    # invert and keep positive
    return (g.max() - g) + 1e-6

def wall(grid_size, x1, y1, x2, y2):
    h, w = grid_size
    K = np.zeros((h, w), float)

    length = int(max(abs(x2 - x1), abs(y2 - y1))) + 1
    xs = np.linspace(x1, x2, length).astype(int)
    ys = np.linspace(y1, y2, length).astype(int)

    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)

    K[ys, xs] = 1.0
    return K

def edge_bias(grid_size, strength=1.0):
    h, w = grid_size
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]

    dist_y = np.minimum(y, h - 1 - y)
    dist_x = np.minimum(x, w - 1 - x)
    dist_to_edge = np.minimum(dist_y, dist_x)

    dist_norm = dist_to_edge / dist_to_edge.max()
    return (1 - dist_norm)**strength

def multi_gaussian(grid_size, centers, sigma):
    h, w = grid_size
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    K = np.zeros((h, w), float)

    for cy, cx in centers:
        dist2 = (y - cy)**2 + (x - cx)**2
        K += np.exp(-dist2 / (2 * sigma**2))

    return K

import numpy as np

def multi_negative_gaussian(grid_size, centers, sigma):
    """
    centers: list of (y, x) tuples
    sigma: float
    """
    h, w = grid_size
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]

    # Build positive Gaussians first
    G = np.zeros((h, w), float)
    for cy, cx in centers:
        dist2 = (y - cy)**2 + (x - cx)**2
        G += np.exp(-dist2 / (2 * sigma**2))

    # Invert the combined field
    # Add epsilon to avoid zeros
    G_max = G.max() if G.max() > 0 else 1.0
    K = (G_max - G) + 1e-6

    return K

def uniform_kernel(grid_size):
    h, w = grid_size
    return np.ones((h, w), float)