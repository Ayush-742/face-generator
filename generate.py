"""Generate faces from a trained generator.

  python generate.py --weights checkpoints/generator.pth --n 64 --seed 7
  python generate.py --weights checkpoints/generator.pth --interpolate --seed 1 --seed2 2
"""
import argparse

from utils import generate, load_generator, noise_from_seed, slerp, to_pil_grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="checkpoints/generator.pth")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seed2", type=int, default=43)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--interpolate", action="store_true")
    p.add_argument("--out", default="generated.png")
    args = p.parse_args()

    g = load_generator(args.weights)
    if args.interpolate:
        z = slerp(noise_from_seed(args.seed)[0], noise_from_seed(args.seed2)[0], args.steps)
        img = to_pil_grid(generate(g, z), nrow=args.steps)
    else:
        img = to_pil_grid(generate(g, noise_from_seed(args.seed, args.n)), nrow=8)
    img.save(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
