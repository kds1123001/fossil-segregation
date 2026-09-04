import numpy as np
from skimage.measure import marching_cubes


def mask_to_mesh(mask, spacing=(1.0, 1.0, 1.0), smooth_step=0.5):
    verts, faces, normals, _ = marching_cubes(mask.astype(np.float32), level=smooth_step, spacing=spacing)
    return verts, faces, normals


def write_obj(path, verts, faces, normals=None):
    with open(path, "w") as f:
        for i, v in enumerate(verts):
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        if normals is not None:
            for n in normals:
                f.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
        for face in faces:
            idx = face + 1
            if normals is not None:
                f.write(f"f {idx[0]}//{idx[0]} {idx[1]}//{idx[1]} {idx[2]}//{idx[2]}\n")
            else:
                f.write(f"f {idx[0]} {idx[1]} {idx[2]}\n")
