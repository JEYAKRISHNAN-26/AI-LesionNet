"""AI-LesionNet: 3D MRI context/uncertainty segmentation implementation.
Tensor order is [batch, modality, depth, height, width].
"""
import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels_in, channels_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(channels_out), nn.ReLU(inplace=True),
            nn.Conv3d(channels_out, channels_out, 3, padding=1, bias=False),
            nn.BatchNorm3d(channels_out), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ContextualFusion(nn.Module):
    """Bottleneck patch tokens with 6-layer MHSA and modality cross-attention."""
    def __init__(self, channels=512, modality_channels=8, embed_dim=256,
                 heads=8, layers=6, patch=4, transformer_dropout=0.1):
        super().__init__()
        self.patch = patch
        self.tokenize = nn.Conv3d(channels, embed_dim, patch, stride=patch)
        self.modality_tokenize = nn.Conv3d(modality_channels, embed_dim, patch, stride=patch)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim * 4,
            dropout=transformer_dropout, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(enc_layer, num_layers=layers,
                                             enable_nested_tensor=False)
        self.cross_attention = nn.MultiheadAttention(embed_dim, heads, dropout=transformer_dropout,
                                                      batch_first=True)
        self.cross_norm = nn.LayerNorm(embed_dim)
        self.restore = nn.ConvTranspose3d(embed_dim, embed_dim, patch, stride=patch)
        self.combine = nn.Conv3d(channels + embed_dim, channels, kernel_size=1)

    def forward(self, bottleneck, modality_maps):
        # The transformer operates on a downsampled bottleneck, not millions of voxels.
        b, _, d, h, w = bottleneck.shape
        p = self.patch
        padded = F.pad(bottleneck, (0, (-w) % p, 0, (-h) % p, 0, (-d) % p))
        tokens_map = self.tokenize(padded)
        td, th, tw = tokens_map.shape[2:]
        q = tokens_map.flatten(2).transpose(1, 2)
        q = self.context(q)
        modality_tokens = []
        for map_i in modality_maps:
            resized = F.adaptive_avg_pool3d(map_i, bottleneck.shape[2:])
            resized = F.pad(resized, (0, (-w) % p, 0, (-h) % p, 0, (-d) % p))
            modality_tokens.append(self.modality_tokenize(resized).flatten(2).transpose(1, 2))
        kv = torch.cat(modality_tokens, dim=1)
        cross, _ = self.cross_attention(q, kv, kv, need_weights=False)
        fused = self.cross_norm(q + cross)
        restored = fused.transpose(1, 2).reshape(b, -1, td, th, tw)
        restored = self.restore(restored)[..., :d, :h, :w]
        return self.combine(torch.cat((bottleneck, restored), dim=1))


class AILesionNet(nn.Module):
    """3-modality 3D CNN/transformer U-Net with stochastic-dropout inference."""
    def __init__(self, base_channels=32, embed_dim=256, heads=8, layers=6,
                 patch=4, dropout=0.3, transformer_dropout=0.1,
                 use_fusion=True):
        super().__init__()
        self.use_fusion = use_fusion
        if embed_dim % heads:
            raise ValueError('embed_dim must be divisible by attention heads')
        c = base_channels
        mc = max(4, c // 4)
        self.modal_stems = nn.ModuleList([ConvBlock(1, mc) for _ in range(3)])
        self.enc1 = ConvBlock(3 * mc, c)
        self.enc2 = ConvBlock(c, c * 2, stride=2)
        self.enc3 = ConvBlock(c * 2, c * 4, stride=2)
        self.enc4 = ConvBlock(c * 4, c * 8, stride=2)
        self.bottleneck = ConvBlock(c * 8, c * 16)
        self.fusion = ContextualFusion(c * 16, mc, embed_dim, heads, layers, patch,
                                       transformer_dropout)
        self.stochastic = nn.Dropout3d(p=dropout)
        self.up3 = nn.ConvTranspose3d(c * 16, c * 4, 2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose3d(c * 2, c, 2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)
        self.out = nn.Conv3d(c, 1, kernel_size=1)

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != 3:
            raise ValueError('Expected MRI [B, 3, D, H, W] (T1, T2, FLAIR)')
        mods = [stem(x[:, i:i+1]) for i, stem in enumerate(self.modal_stems)]
        e1 = self.enc1(torch.cat(mods, dim=1))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        z = self.stochastic(self.bottleneck(e4))
        if self.use_fusion:
            z = self.fusion(z, mods)
        z = self.dec3(torch.cat([self.up3(z), e3], dim=1))
        z = self.dec2(torch.cat([self.up2(z), e2], dim=1))
        # The final high-resolution skip conserves fine lesion boundaries.
        z = self.dec1(torch.cat([self.up1(z), e1], dim=1))
        return self.out(self.stochastic(z))  # Logits, not probabilities


def activate_mc_dropout(model):
    """Keep batch-normalization fixed; enable only dropout for MC inference."""
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


@torch.no_grad()
def predict_mc(model, x, iterations=20):
    if iterations < 1:
        raise ValueError('iterations must be >= 1')
    activate_mc_dropout(model)
    samples = torch.stack([torch.sigmoid(model(x)) for _ in range(iterations)])
    return samples.mean(0), samples.var(0, unbiased=False)
