"""Train a DCGAN on face images.

Example (Kaggle / Colab):
  python train.py --data_dir /kaggle/input/celeba-dataset --epochs 30 --max_images 50000
Quick smoke test on any machine:
  python train.py --data_dir data --epochs 1 --max_images 2000
"""
import argparse
import csv
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

from models import NZ, Discriminator, Generator, weights_init


class FaceDataset(Dataset):
    """Finds every image under data_dir (any folder layout), resizes and normalizes."""

    def __init__(self, root, image_size=64, max_images=None, seed=42):
        exts = {".jpg", ".jpeg", ".png"}
        files = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in exts)
        if not files:
            raise SystemExit(f"No images found under '{root}'. Check --data_dir.")
        if max_images and len(files) > max_images:
            files = random.Random(seed).sample(files, max_images)
        self.files = files
        self.tf = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1, 1]
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.tf(Image.open(self.files[i]).convert("RGB"))


def smooth(x, k=50):
    if len(x) < k:
        return np.array(x)
    return np.convolve(x, np.ones(k) / k, mode="valid")


def save_loss_plot(lossG, lossD, path):
    plt.figure(figsize=(9, 4))
    plt.plot(lossG, alpha=0.2, color="tab:blue")
    plt.plot(lossD, alpha=0.2, color="tab:orange")
    plt.plot(smooth(lossG), color="tab:blue", label="Generator")
    plt.plot(smooth(lossD), color="tab:orange", label="Discriminator")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("DCGAN training losses (smoothed)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", default="out")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--image_size", type=int, default=64, help="Architecture is built for 64")
    p.add_argument("--max_images", type=int, default=50000)
    p.add_argument("--real_label", type=float, default=0.9, help="One-sided label smoothing")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out = Path(args.out_dir)
    (out / "samples").mkdir(parents=True, exist_ok=True)

    ds = FaceDataset(args.data_dir, args.image_size, args.max_images, args.seed)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    print(f"Training on {len(ds)} images, {len(loader)} iterations per epoch")

    # Real images for side-by-side comparison in the README
    save_image(
        torch.stack([ds[i] for i in range(min(64, len(ds)))]),
        out / "real_samples.png",
        nrow=8,
        normalize=True,
        value_range=(-1, 1),
    )

    netG, netD = Generator().to(device), Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCEWithLogitsLoss()
    optD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    fixed_noise = torch.randn(64, NZ, device=device)  # same z every epoch -> visible progress

    lossG_hist, lossD_hist, rows = [], [], []

    for epoch in range(1, args.epochs + 1):
        ep_g, ep_d, ep_dx, ep_dgz = [], [], [], []
        for real in loader:
            real = real.to(device, non_blocking=True)
            b = real.size(0)

            # ---- Discriminator: maximize log D(x) + log(1 - D(G(z))) ----
            netD.zero_grad()
            out_real = netD(real)
            loss_real = criterion(out_real, torch.full((b,), args.real_label, device=device))
            fake = netG(torch.randn(b, NZ, device=device))
            out_fake = netD(fake.detach())
            loss_fake = criterion(out_fake, torch.zeros(b, device=device))
            lossD = loss_real + loss_fake
            lossD.backward()
            optD.step()

            # ---- Generator: maximize log D(G(z)) ----
            netG.zero_grad()
            lossG = criterion(netD(fake), torch.ones(b, device=device))
            lossG.backward()
            optG.step()

            lossG_hist.append(lossG.item())
            lossD_hist.append(lossD.item())
            ep_g.append(lossG.item())
            ep_d.append(lossD.item())
            ep_dx.append(torch.sigmoid(out_real).mean().item())
            ep_dgz.append(torch.sigmoid(out_fake).mean().item())

        row = {
            "epoch": epoch,
            "loss_D": float(np.mean(ep_d)),
            "loss_G": float(np.mean(ep_g)),
            "D(x)": float(np.mean(ep_dx)),
            "D(G(z))": float(np.mean(ep_dgz)),
        }
        rows.append(row)
        print(
            f"Epoch {epoch}/{args.epochs} | loss_D {row['loss_D']:.3f} | "
            f"loss_G {row['loss_G']:.3f} | D(x) {row['D(x)']:.2f} | D(G(z)) {row['D(G(z))']:.2f}"
        )

        # Save sample grid, generator weights, and logs every epoch
        netG.eval()
        with torch.no_grad():
            samples = netG(fixed_noise).cpu()
        netG.train()
        save_image(
            samples,
            out / "samples" / f"epoch_{epoch:03d}.png",
            nrow=8,
            normalize=True,
            value_range=(-1, 1),
        )
        torch.save(netG.state_dict(), out / "generator.pth")
        with open(out / "losses.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        save_loss_plot(lossG_hist, lossD_hist, out / "loss_curves.png")

    print(f"\nDone. Results in ./{out}/  (generator.pth, samples/, loss_curves.png, losses.csv)")


if __name__ == "__main__":
    main()
