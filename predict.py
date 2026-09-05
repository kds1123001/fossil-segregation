import argparse
import sys
import os
#blehhhh





import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fossilseg.model import UNet3D
from fossilseg.infer import sliding_window_infer
from fossilseg.mesh import mask_to_mesh, write_obj
from fossilseg.visualize import plot_orthogonal_slices


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--volume", required=True)
    p.add_argument("--out_dir", default="predictions")
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--base_ch", type=int, default=16)
    p.add_argument("--mesh", action="store_true")
    p.add_argument("--mask", default=None, help="ground-truth mask for side-by-side viz")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    model = UNet3D(base_ch=args.base_ch).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    vol = np.load(args.volume)
    pred, probs = sliding_window_infer(model, vol, patch_size=args.patch_size,
                                        overlap=args.overlap, device=device)

    stem = os.path.splitext(os.path.basename(args.volume))[0]
    np.save(os.path.join(args.out_dir, f"{stem}_pred.npy"), pred)

    gt = np.load(args.mask) if args.mask else None
    plot_orthogonal_slices(vol, mask=gt, pred=pred, out_path=os.path.join(args.out_dir, f"{stem}_slices.png"))

    if args.mesh:
        verts, faces, normals = mask_to_mesh(pred)
        write_obj(os.path.join(args.out_dir, f"{stem}_mesh.obj"), verts, faces, normals)

    print(f"predicted voxels: {pred.sum()} / {pred.size} ({100*pred.sum()/pred.size:.3f}%)")


if __name__ == "__main__":
    main()
