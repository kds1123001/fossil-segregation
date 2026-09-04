import numpy as np
import torch
import torch.nn.functional as F


def gaussian_weight(patch_size, sigma_scale=0.125):
    coords = [np.linspace(-1, 1, s) for s in patch_size]
    grids = np.meshgrid(*coords, indexing="ij")
    dist2 = sum(g ** 2 for g in grids)
    sigma = sigma_scale
    w = np.exp(-dist2 / (2 * sigma ** 2))
    return w.astype(np.float32)


def sliding_window_infer(model, volume, patch_size=64, overlap=0.5, device="cpu", n_classes=2, batch_size=4):
    model.eval()
    shape = volume.shape
    stride = [max(1, int(p * (1 - overlap))) for p in [patch_size] * 3]
    ps = [patch_size] * 3

    pad = [max(0, ps[d] - shape[d]) for d in range(3)]
    vol_p = np.pad(volume, [(0, pad[d]) for d in range(3)], mode="reflect")
    padded_shape = vol_p.shape

    starts = []
    for d in range(3):
        s = list(range(0, padded_shape[d] - ps[d] + 1, stride[d]))
        if not s or s[-1] != padded_shape[d] - ps[d]:
            s.append(padded_shape[d] - ps[d])
        starts.append(s)

    weight_patch = gaussian_weight(ps)
    acc = np.zeros((n_classes,) + padded_shape, dtype=np.float32)
    weight_sum = np.zeros(padded_shape, dtype=np.float32)

    coords_list = [(z, y, x) for z in starts[0] for y in starts[1] for x in starts[2]]

    with torch.no_grad():
        for i in range(0, len(coords_list), batch_size):
            batch_coords = coords_list[i:i + batch_size]
            patches = np.stack([
                vol_p[z:z + ps[0], y:y + ps[1], x:x + ps[2]] for (z, y, x) in batch_coords
            ])
            inp = torch.from_numpy(patches).unsqueeze(1).float().to(device)
            logits = model(inp)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            for j, (z, y, x) in enumerate(batch_coords):
                acc[:, z:z + ps[0], y:y + ps[1], x:x + ps[2]] += probs[j] * weight_patch
                weight_sum[z:z + ps[0], y:y + ps[1], x:x + ps[2]] += weight_patch

    acc /= np.clip(weight_sum, 1e-6, None)
    acc = acc[:, :shape[0], :shape[1], :shape[2]]
    pred = np.argmax(acc, axis=0).astype(np.uint8)
    return pred, acc
