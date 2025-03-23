import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from datetime import datetime
from typing import Dict, Tuple, Optional


def denormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize a normalized image tensor to [0,1] range."""
    # Convert from tensor (C,H,W) to numpy (H,W,C)
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy().transpose(1, 2, 0)

    # Denormalize
    img = img * np.array(std) + np.array(mean)
    return np.clip(img, 0, 1)


def visualize_batch(
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
        output_dir: str = './visualizations',
        epoch: Optional[int] = None,
        batch_idx: Optional[int] = None,
        max_samples: int = 4,
        save_fig: bool = True
) -> plt.Figure:
    """
    Visualize model predictions on a batch of data during training.

    Args:
        model: The model being trained
        batch: Dictionary containing 'image' and 'mask' tensors
        device: Device for inference
        output_dir: Directory to save visualizations
        epoch: Current epoch (for filename)
        batch_idx: Current batch index (for filename)
        max_samples: Maximum number of samples to visualize
        save_fig: Whether to save the figure to disk

    Returns:
        The matplotlib figure object
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Set model to evaluation mode temporarily
    model.eval()

    # Get data from batch
    images = batch['image'].to(device)
    masks = batch['mask'].detach().cpu()

    # Limit to max_samples
    if images.shape[0] > max_samples:
        images = images[:max_samples]
        masks = masks[:max_samples]

    num_samples = images.shape[0]

    # Make predictions
    with torch.no_grad():
        outputs = model(images)
        if outputs.shape[1] == 1:  # Binary segmentation
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
        else:  # Multi-class segmentation
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1, keepdim=True)

    # Move predictions to CPU
    preds = preds.detach().cpu()

    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    # Handle case with only one sample
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    # Plot each sample
    for i in range(num_samples):
        # Original image
        img = denormalize(images[i].cpu())
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')

        # Ground truth mask
        axes[i, 1].imshow(masks[i].squeeze().numpy(), cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        # Predicted mask
        axes[i, 2].imshow(preds[i].squeeze().numpy(), cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')

    plt.tight_layout()

    # Save figure if requested
    if save_fig:
        # Create filename with timestamp if epoch/batch not provided
        if epoch is not None and batch_idx is not None:
            filename = f'vis_epoch_{epoch}_batch_{batch_idx}.png'
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'vis_{timestamp}.png'

        plt.savefig(os.path.join(output_dir, filename))

    # Set model back to training mode
    model.train()

    return fig


def visualize_training_progress(
        model: torch.nn.Module,
        train_batch: Dict[str, torch.Tensor],
        val_batch: Dict[str, torch.Tensor],
        device: torch.device,
        output_dir: str = './visualizations',
        epoch: Optional[int] = None,
        global_step: Optional[int] = None
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Visualize model predictions on both training and validation data.

    Args:
        model: The model being trained
        train_batch: Dictionary containing training data
        val_batch: Dictionary containing validation data
        device: Device for inference
        output_dir: Directory to save visualizations
        epoch: Current epoch
        global_step: Global training step

    Returns:
        Tuple of (train_figure, val_figure)
    """
    # Create train visualization
    train_fig = visualize_batch(
        model=model,
        batch=train_batch,
        device=device,
        output_dir=os.path.join(output_dir, 'train'),
        epoch=epoch,
        batch_idx=global_step,
        max_samples=2,
        save_fig=True
    )

    # Create validation visualization
    val_fig = visualize_batch(
        model=model,
        batch=val_batch,
        device=device,
        output_dir=os.path.join(output_dir, 'validation'),
        epoch=epoch,
        batch_idx=global_step,
        max_samples=2,
        save_fig=True
    )

    return train_fig, val_fig
