import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Callable, Optional, Tuple, Union


def get_transforms(
        mode: str = 'train',
        mean: Optional[list] = None,
        std: Optional[list] = None,
        target_size: int = 224,
        apply_grayscale: bool = True,
        grayscale_prob: float = 0.2
) -> Callable:
    """
    Get transforms for AI inpainting detection tasks.

    Args:
        mode: 'train' or 'val'
        mean: Mean values for normalization (defaults to ImageNet means)
        std: Standard deviation values for normalization (defaults to ImageNet stds)
        target_size: Target size for resizing (square images)
        apply_grayscale: Whether to apply grayscale transformation
        grayscale_prob: Probability of applying grayscale transformation

    Returns:
        Albumentations transform composition
    """
    # Default to ImageNet means and stds if not provided
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    if mode == 'train':
        # Base transforms with corrected RandomSizedCrop parameters
        base_transforms = [
            # Random 90-degree rotations
            A.RandomRotate90(p=0.5),

            # Random flipping
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),

            # Correctly parameterized RandomSizedCrop
            # In get_training_transforms:
            # Be careful with RandomSizedCrop for segmentation tasks
            A.RandomSizedCrop(
                min_max_height=(int(target_size * 0.8), target_size),  # Less aggressive
                size=(target_size, target_size),
                w2h_ratio=1.0,
                p=0.3  # Lower probability
            ),

            # Resize to target size (square)
            A.Resize(height=target_size, width=target_size),
        ]

        # Grayscale transform that can be enabled/disabled
        grayscale_transforms = []
        if apply_grayscale:
            grayscale_transforms = [
                A.ToGray(p=grayscale_prob),
            ]

        # Normalization and tensor conversion
        final_transforms = [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]

        # Combine all transforms
        transforms = A.Compose(base_transforms + grayscale_transforms + final_transforms)
    else:
        # Validation transforms (no augmentation, just resize, normalize and convert)
        transforms = A.Compose([
            # Resize to target size (square)
            A.Resize(height=target_size, width=target_size),

            # Normalization and tensor conversion
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    return transforms


def get_training_transforms(
        mean: Optional[list] = None,
        std: Optional[list] = None,
        target_size: int = 224,
        apply_grayscale: bool = True,
        grayscale_prob: float = 0.2
) -> Callable:
    """
    Get transforms for training data with augmentations.

    Args:
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        target_size: Target size for resizing (square images)
        apply_grayscale: Whether to apply grayscale transformation
        grayscale_prob: Probability of applying grayscale transformation

    Returns:
        Albumentations transform composition
    """
    return get_transforms(
        mode='train',
        mean=mean,
        std=std,
        target_size=target_size,
        apply_grayscale=apply_grayscale,
        grayscale_prob=grayscale_prob
    )


def get_validation_transforms(
        mean: Optional[list] = None,
        std: Optional[list] = None,
        target_size: int = 224
) -> Callable:
    """
    Get transforms for validation data (no augmentation).

    Args:
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        target_size: Target size for resizing (square images)

    Returns:
        Albumentations transform composition
    """
    return get_transforms(mode='val', mean=mean, std=std, target_size=target_size)
