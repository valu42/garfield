import os
import torch
import glob
from typing import Dict, Any, Optional, Union, Tuple

from src.configs.config import Config


def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_loss: float,
        val_metric: float,
        global_step: int,
        config: Config,
        scheduler: Optional[Any] = None,
        is_best: bool = False,
        filename: str = 'checkpoint.pth'
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch number
        val_loss: Validation loss
        val_metric: Validation metric (e.g., Dice coefficient)
        global_step: Global training step
        config: Configuration object
        scheduler: Learning rate scheduler (optional)
        is_best: Whether this is the best model so far
        filename: Name of the checkpoint file
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Prepare checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_metric': val_metric,
        'global_step': global_step,
        'config': config.__dict__,  # Save configuration
    }

    # Add scheduler state if it exists
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # Save checkpoint
    checkpoint_path = os.path.join(config.checkpoint_dir, filename)
    torch.save(checkpoint, checkpoint_path)

    # Save as latest checkpoint
    latest_path = os.path.join(config.checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)

    # Save as best model if is_best
    if is_best:
        best_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"Saved best model with validation metric: {val_metric:.4f}")


def load_checkpoint(
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        override_lr: float = None
) -> Dict[str, Any]:
    """
    Load model and training state from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Learning rate scheduler to load state into (optional)
        device: Device to load the model to (optional)
        override_lr: If provided, override the learning rate with this value (optional)

    Returns:
        Dictionary containing checkpoint information
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load checkpoint on CPU to avoid GPU memory issues
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])

    # Move model to device if specified
    if device is not None:
        model.to(device)

    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Override learning rate with 1e-5 if specified
        if override_lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = override_lr
            print(f"Learning rate overridden to {override_lr}")

    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}, Validation Metric: {checkpoint.get('val_metric', 'N/A'):.4f}")

    return checkpoint


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in the checkpoint directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to the latest checkpoint file or None if not found
    """
    if not os.path.exists(checkpoint_dir):
        return None

    # First check if latest_checkpoint.pth exists
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    if os.path.exists(latest_path):
        return latest_path

    # Otherwise find the latest by modification time
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    if not checkpoint_files:
        return None

    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    return latest_checkpoint


def find_best_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the best checkpoint in the checkpoint directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to the best checkpoint file or None if not found
    """
    best_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_path):
        return best_path
    return None


def resume_from_checkpoint(
        config: Config,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        load_best: bool = False
) -> Tuple[int, float, float, int]:
    """
    Resume training from the latest or best checkpoint.

    Args:
        config: Configuration object
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Learning rate scheduler to load state into (optional)
        load_best: Whether to load the best checkpoint instead of latest

    Returns:
        Tuple of (start_epoch, best_val_loss, best_val_metric, global_step)
    """
    # Set device
    device = torch.device(config.device)

    # Find checkpoint
    if load_best:
        checkpoint_path = find_best_checkpoint(config.checkpoint_dir)
        if checkpoint_path is None:
            print("Best checkpoint not found. Starting training from scratch.")
            return 0, float('inf'), 0.0, 0
    else:
        checkpoint_path = find_latest_checkpoint(config.checkpoint_dir)
        if checkpoint_path is None:
            print("No checkpoint found. Starting training from scratch.")
            return 0, float('inf'), 0.0, 0

    # Load checkpoint with hardcoded learning rate override
    checkpoint = load_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        device,
        override_lr=1e-5  # Hardcoded learning rate
    )

    # Get resuming information
    start_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch
    best_val_loss = checkpoint.get('val_loss', float('inf'))
    best_val_metric = checkpoint.get('val_metric', 0.0)
    global_step = checkpoint.get('global_step', 0)

    print(f"Resuming from epoch {start_epoch}, global step {global_step}")
    return start_epoch, best_val_loss, best_val_metric, global_step
