import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, 3, padding=1),
        nn.InstanceNorm3d(out_ch, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Conv3d(out_ch, out_ch, 3, padding=1),
        nn.InstanceNorm3d(out_ch, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
    )


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = conv_block(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        diffs = [s - x_ for s, x_ in zip(skip.shape[2:], x.shape[2:])]
        if any(diffs):
            x = F.pad(x, [diffs[2] // 2, diffs[2] - diffs[2] // 2,
                          diffs[1] // 2, diffs[1] - diffs[1] // 2,
                          diffs[0] // 2, diffs[0] - diffs[0] // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, base_ch=16):
        super().__init__()
        c = base_ch
        self.stem = conv_block(in_channels, c)
        self.d1 = Down(c, c * 2)
        self.d2 = Down(c * 2, c * 4)
        self.d3 = Down(c * 4, c * 8)
        self.d4 = Down(c * 8, c * 16)

        self.u1 = Up(c * 16, c * 8, c * 8)
        self.u2 = Up(c * 8, c * 4, c * 4)
        self.u3 = Up(c * 4, c * 2, c * 2)
        self.u4 = Up(c * 2, c, c)

        self.head = nn.Conv3d(c, n_classes, 1)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)

        y = self.u1(x4, x3)
        y = self.u2(y, x2)
        y = self.u3(y, x1)
        y = self.u4(y, x0)
        return self.head(y)


if __name__ == "__main__":
    net = UNet3D(base_ch=8)
    x = torch.randn(1, 1, 64, 64, 64)
    out = net(x)
    print(out.shape, sum(p.numel() for p in net.parameters()))
