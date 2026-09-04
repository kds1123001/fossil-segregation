import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fossilseg.synth import generate_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--shape", type=int, nargs=3, default=[96, 96, 96])
    p.add_argument("--seed0", type=int, default=0)
    args = p.parse_args()

    n = generate_dataset(args.n, args.out_dir, shape=tuple(args.shape), seed0=args.seed0)
    print(f"wrote {n} volumes to {args.out_dir}")


if __name__ == "__main__":
    main()
