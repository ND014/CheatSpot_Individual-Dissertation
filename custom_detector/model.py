import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1)
        )
    def forward(self, x): return self.block(x)


class CheatNet(nn.Module):
    """
    Custom single-shot object detector
    Input:  (B, 3, 416, 416)
    Output: (B, 6, 13, 13)
            6 = [tx, ty, tw, th, objectness, class_score]
    """
    def __init__(self, num_classes=1):
        super().__init__()
        self.num_classes = num_classes

        # ── Backbone ──────────────────────────────────
        self.backbone = nn.Sequential(
            ConvBlock(3,   32,  3, 1, 1),   # 416→416
            nn.MaxPool2d(2, 2),              # 416→208

            ConvBlock(32,  64,  3, 1, 1),
            nn.MaxPool2d(2, 2),              # 208→104

            ConvBlock(64,  128, 3, 1, 1),
            ConvBlock(128, 64,  1, 1, 0),    # 1x1 conv — bottleneck
            ConvBlock(64,  128, 3, 1, 1),
            nn.MaxPool2d(2, 2),              # 104→52

            ConvBlock(128, 256, 3, 1, 1),
            ConvBlock(256, 128, 1, 1, 0),
            ConvBlock(128, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),              # 52→26

            ConvBlock(256, 512, 3, 1, 1),
            ConvBlock(512, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),              # 26→13
        )

        # ── Neck ──────────────────────────────────────
        self.neck = nn.Sequential(
            ConvBlock(512, 1024, 3, 1, 1),
            ConvBlock(1024, 512, 1, 1, 0),
            ConvBlock(512, 1024, 3, 1, 1),
            ConvBlock(1024, 512, 1, 1, 0),
            ConvBlock(512, 1024, 3, 1, 1),
        )

        self.head = nn.Conv2d(1024, 5 + num_classes, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x   # shape: (B, 6, 13, 13)


if __name__ == '__main__':
    model  = CheatNet(num_classes=1)
    dummy  = torch.randn(2, 3, 416, 416)
    output = model(dummy)
    print(f"✅ Output shape: {output.shape}")
    total  = sum(p.numel() for p in model.parameters())
    print(f"✅ Total params: {total:,}")
