"""Helpers shared by generate.py and the Streamlit app."""
import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid

from models import NZ, Generator


def load_generator(path, device="cpu"):
    g = Generator()
    g.load_state_dict(torch.load(path, map_location=device))
    return g.to(device).eval()


def noise_from_seed(seed, n=1):
    """Reproducible latent vectors: same seed -> same faces."""
    gen = torch.Generator().manual_seed(int(seed))
    return torch.randn(n, NZ, generator=gen)


def slerp(z1, z2, steps):
    """Spherical interpolation between two latent vectors of shape (NZ,)."""
    t = torch.linspace(0, 1, steps).view(-1, 1)
    omega = torch.acos(((z1 / z1.norm()) * (z2 / z2.norm())).sum().clamp(-1, 1))
    so = torch.sin(omega)
    if so.abs() < 1e-6:  # identical vectors
        return z1 * (1 - t) + z2 * t
    return (torch.sin((1 - t) * omega) / so) * z1 + (torch.sin(t * omega) / so) * z2


@torch.no_grad()
def generate(g, z):
    return g(z.to(next(g.parameters()).device)).cpu()


def to_pil_grid(images, nrow=8, scale=2):
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1), padding=2)
    arr = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.BICUBIC)
    return img
