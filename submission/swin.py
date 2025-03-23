import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union
import timm
from timm.models.swin_transformer import SwinTransformer


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, use_batchnorm: bool = True):
        super().__init__()
        conv_in = in_channels + skip_channels if skip_channels > 0 else in_channels
        self.conv1 = nn.Conv2d(conv_in, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        if skip is not None:
            # Upsample to match skip spatial size and concatenate.
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        else:
            # Upsample spatially.
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = F.relu(x, inplace=True)
        return x


class SwinEncoderWrapper(nn.Module):
    """
    A simplified encoder that runs the input sequentially through the Swin Transformer's patch
    embedding and layers. This version converts the patch embedding output into [B, H, W, C] format,
    which the Swin layers expect, and then converts intermediate features to [B, C, H, W] for the decoder.
    """
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 192,
        depths: List[int] = [2, 2, 18, 2],
        num_heads: List[int] = [6, 12, 24, 48],
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        norm_layer: nn.Module = nn.LayerNorm,
        patch_norm: bool = True,
        pretrained: Optional[Union[bool, str]] = None,
    ):
        super().__init__()
        print(f"PRETRAINED::: ", pretrained)
        if pretrained is True:
            # Use timm's built-in function to create a model with default pretrained weights.
            self.swin = timm.create_model(
                'swin_large_patch4_window7_224',
                pretrained=True,
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
            )
            print(f"pretrained!!!")
        else:
            self.swin = SwinTransformer(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                num_classes=1000,
                embed_dim=embed_dim,
                depths=depths,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=None,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                norm_layer=norm_layer,
                patch_norm=patch_norm,
            )
            if isinstance(pretrained, str):
                try:
                    state_dict = torch.load(pretrained, map_location='cpu')
                    if 'model' in state_dict:
                        state_dict = state_dict['model']
                    self.swin.load_state_dict(state_dict, strict=False)
                    print(f"Loaded pretrained weights from {pretrained}")
                except Exception as e:
                    print(f"Warning: Failed to load pretrained weights: {e}")

        # Remove classification head.
        self.swin.head = nn.Identity()

        # Expected feature channels per stage.
        self.feature_channels = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        B, _, _, _ = x.shape

        # 1. Patch embedding.
        x = self.swin.patch_embed(x)
        # If patch_embed returns [B, L, C], reshape it to [B, H, W, C].
        if x.dim() == 3:
            B, L, C = x.shape
            h = w = int(L ** 0.5)
            x = x.view(B, h, w, C)
        # Append patch embedding output as a feature (converted to channels-first).
        features.append(x.permute(0, 3, 1, 2))

        # 2. Process through each layer sequentially.
        for layer in self.swin.layers:
            x = layer(x)  # Each layer expects input shape [B, H, W, C] and outputs the same.
            features.append(x.permute(0, 3, 1, 2))
        return features


class DirectSwinUNet(nn.Module):
    """
    A UNet-like segmentation model using the simplified Swin encoder.
    Assumes the encoder returns 5 feature maps:
      - features[0]: patch embedding output
      - features[1]-features[4]: outputs from successive layers (with features[4] as bottleneck)
    """
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 1,
        embed_dim: int = 192,
        depths: List[int] = [2, 2, 18, 2],
        num_heads: List[int] = [6, 12, 24, 48],
        window_size: int = 8,
        decoder_channels: List[int] = [512, 256, 128, 64, 32],
        use_batchnorm: bool = True,
        pretrained: Optional[Union[bool, str]] = None,
    ):
        super().__init__()
        self.encoder = SwinEncoderWrapper(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            pretrained=pretrained,
        )
        self.feature_channels = self.encoder.feature_channels

        self.decoder_blocks = nn.ModuleList([
            # Decoder block: bottleneck + skip from previous stage.
            DecoderBlock(in_channels=self.feature_channels[3], skip_channels=self.feature_channels[2],
                         out_channels=decoder_channels[0], use_batchnorm=use_batchnorm),
            DecoderBlock(in_channels=decoder_channels[0], skip_channels=self.feature_channels[1],
                         out_channels=decoder_channels[1], use_batchnorm=use_batchnorm),
            DecoderBlock(in_channels=decoder_channels[1], skip_channels=self.feature_channels[0],
                         out_channels=decoder_channels[2], use_batchnorm=use_batchnorm),
            DecoderBlock(in_channels=decoder_channels[2], skip_channels=0,
                         out_channels=decoder_channels[3], use_batchnorm=use_batchnorm),
            DecoderBlock(in_channels=decoder_channels[3], skip_channels=0,
                         out_channels=decoder_channels[4], use_batchnorm=use_batchnorm),
        ])
        self.final_conv = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        features = self.encoder(x)
        # Use the last feature as the bottleneck.
        x = features[-1]
        # Decode with skip connections.
        x = self.decoder_blocks[0](x, features[-2])
        x = self.decoder_blocks[1](x, features[-3])
        x = self.decoder_blocks[2](x, features[-4])
        x = self.decoder_blocks[3](x, None)
        x = self.decoder_blocks[4](x, None)
        if x.shape[2] != H or x.shape[3] != W:
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return self.final_conv(x)


def build_swin_large_unet(
    img_size: int = 256,
    patch_size: int = 4,
    in_chans: int = 3,
    num_classes: int = 1,
    window_size: int = 8,
    decoder_channels: Optional[List[int]] = None,
    use_batchnorm: bool = True,
    pretrained: Optional[Union[bool, str]] = None,
) -> nn.Module:

    print(f"pretrained: {pretrained} I(first)")

    if decoder_channels is None:
        decoder_channels = [512, 256, 128, 64, 32]
    model = DirectSwinUNet(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=window_size,
        decoder_channels=decoder_channels,
        use_batchnorm=use_batchnorm,
        pretrained=pretrained,
    )
    return model


if __name__ == "__main__":
    model = build_swin_large_unet(
        img_size=256,
        patch_size=4,
        in_chans=3,
        num_classes=1,
        window_size=8,
        pretrained=True,  # Set to True to use timm's default pretrained weights.
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = torch.randn(2, 3, 256, 256, device=device)
    from torch.amp import autocast
    with autocast(device_type='cuda'):
        y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
