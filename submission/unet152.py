import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Optional, Union, Tuple


class DecoderBlock(nn.Module):
    """
    Decoder block for UNet architecture.

    Args:
        in_channels: Number of input channels
        skip_channels: Number of skip connection channels (0 for no skip)
        out_channels: Number of output channels
        use_batchnorm: Whether to use batch normalization
    """

    def __init__(
            self,
            in_channels: int,
            skip_channels: int,
            out_channels: int,
            use_batchnorm: bool = True,
    ):
        super().__init__()
        self.skip_channels = skip_channels

        if skip_channels > 0:
            self.conv1 = nn.Conv2d(
                in_channels + skip_channels, out_channels, kernel_size=3, padding=1
            )
        else:
            # If no skip connection, just use in_channels
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1
            )

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.use_batchnorm = use_batchnorm

        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through decoder block."""
        # Upsample x to match skip dimensions or just scale by 2 if no skip
        if self.skip_channels > 0 and skip is not None:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)
        else:
            # Just upsample by a factor of 2
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # First convolution
        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = F.relu(x, inplace=True)

        # Second convolution
        x = self.conv2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = F.relu(x, inplace=True)

        return x


class UNetResNet152(nn.Module):
    """
    UNet with ResNet152 encoder for segmentation tasks.

    Args:
        num_classes: Number of output classes (1 for binary segmentation)
        pretrained: Whether to use pretrained ResNet152 weights
        decoder_channels: List of decoder channel counts
        use_batchnorm: Whether to use batch normalization
    """

    def __init__(
            self,
            num_classes: int = 1,
            pretrained: bool = True,
            decoder_channels: List[int] = [256, 128, 64, 32, 16],
            use_batchnorm: bool = True,
    ):
        super().__init__()

        # Load ResNet152 as encoder
        resnet = models.resnet152(pretrained=pretrained)

        # Define encoder layers individually as proper submodules
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )  # 64 channels
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels

        # Store encoder channels for decoder
        self.encoder_channels = [64, 256, 512, 1024, 2048]

        # Create decoder blocks - with correct channel specifications
        self.decoder_blocks = nn.ModuleList([
            # For bottleneck layer (input 2048, skip 1024, output 256)
            DecoderBlock(
                in_channels=self.encoder_channels[4],  # 2048
                skip_channels=self.encoder_channels[3],  # 1024
                out_channels=decoder_channels[0],  # 256
                use_batchnorm=use_batchnorm,
            ),
            # For second decoder (input 256, skip 512, output 128)
            DecoderBlock(
                in_channels=decoder_channels[0],  # 256
                skip_channels=self.encoder_channels[2],  # 512
                out_channels=decoder_channels[1],  # 128
                use_batchnorm=use_batchnorm,
            ),
            # For third decoder (input 128, skip 256, output 64)
            DecoderBlock(
                in_channels=decoder_channels[1],  # 128
                skip_channels=self.encoder_channels[1],  # 256
                out_channels=decoder_channels[2],  # 64
                use_batchnorm=use_batchnorm,
            ),
            # For fourth decoder (input 64, skip 64, output 32)
            DecoderBlock(
                in_channels=decoder_channels[2],  # 64
                skip_channels=self.encoder_channels[0],  # 64
                out_channels=decoder_channels[3],  # 32
                use_batchnorm=use_batchnorm,
            ),
            # For last decoder (input 32, no skip, output 16)
            DecoderBlock(
                in_channels=decoder_channels[3],  # 32
                skip_channels=0,  # No skip for the last decoder
                out_channels=decoder_channels[4],  # 16
                use_batchnorm=use_batchnorm,
            ),
        ])

        # Final classification layer
        self.final_conv = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through UNet model."""
        # Store encoder outputs for skip connections
        encoder_outputs = []

        # Layer 0 (initial conv + bn + relu)
        x0 = self.layer0(x)
        encoder_outputs.append(x0)  # [0]: 64 channels

        # Apply maxpool before layer1
        x = self.maxpool(x0)

        # Run through remaining encoder layers
        x1 = self.layer1(x)
        encoder_outputs.append(x1)  # [1]: 256 channels

        x2 = self.layer2(x1)
        encoder_outputs.append(x2)  # [2]: 512 channels

        x3 = self.layer3(x2)
        encoder_outputs.append(x3)  # [3]: 1024 channels

        x4 = self.layer4(x3)
        encoder_outputs.append(x4)  # [4]: 2048 channels

        # Process decoder with skip connections
        # Start with the last encoder output
        x = encoder_outputs[4]  # 2048 channels

        # Process through decoder blocks in sequence
        x = self.decoder_blocks[0](x, encoder_outputs[3])  # 2048 -> 256 with 1024 skip
        x = self.decoder_blocks[1](x, encoder_outputs[2])  # 256 -> 128 with 512 skip
        x = self.decoder_blocks[2](x, encoder_outputs[1])  # 128 -> 64 with 256 skip
        x = self.decoder_blocks[3](x, encoder_outputs[0])  # 64 -> 32 with 64 skip
        x = self.decoder_blocks[4](x, None)  # 32 -> 16 with no skip (upsamples by 2)

        # Final classification
        output = self.final_conv(x)

        return output


def build_unet(
        num_classes: int = 1,
        pretrained: bool = True,
        decoder_channels: Optional[List[int]] = None,
        use_batchnorm: bool = True,
) -> nn.Module:
    """
    Build UNet model with ResNet152 encoder.

    Args:
        num_classes: Number of output classes (1 for binary segmentation)
        pretrained: Whether to use pretrained ResNet152 weights
        decoder_channels: List of decoder channel counts
        use_batchnorm: Whether to use batch normalization

    Returns:
        UNet model with ResNet152 encoder
    """
    if decoder_channels is None:
        decoder_channels = [256, 128, 64, 32, 16]

    model = UNetResNet152(
        num_classes=num_classes,
        pretrained=pretrained,
        decoder_channels=decoder_channels,
        use_batchnorm=use_batchnorm,
    )

    return model
