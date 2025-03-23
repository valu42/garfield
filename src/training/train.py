import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.unet152 import build_unet
from src.models.attention_unet import build_attention_unet
from src.models.swin import build_swin_large_unet
from src.models.swin2 import build_swin2_large_unet
from src.data.dataset import Dataset
from src.data.transforms import get_training_transforms, get_validation_transforms
from src.training.trainer import Trainer
from src.configs.config import Config
from src.utils.checkpoint_utils import resume_from_checkpoint


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to the logits for probabilities
        probs = torch.sigmoid(logits)

        # Flatten the predictions and targets
        batch_size = logits.size(0)
        probs = probs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)

        # Calculate intersection and union
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # Return Dice loss (1 - Dice coefficient)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        combined_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return combined_loss


def main():
    """Main training function."""
    # Get configuration
    config = Config()

    # Set device
    device = torch.device(config.device)
    print(f"Using device: {device}")

    # Create datasets
    train_dataset = Dataset(
        data_dir=config.data_dir,
        split_file=config.split_file_train,
        transforms=get_training_transforms(target_size=256)
    )

    val_dataset = Dataset(
        data_dir=config.data_dir,
        split_file=config.split_file_val,
        transforms=get_validation_transforms(target_size=256)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if str(config.device) == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if str(config.device) == 'cuda' else False
    )

    # Create model
    #model = build_swin2_large_unet(
    #   img_size = 256,
    #   num_classes=1,
    #   pretrained=True
    #)

    model = build_unet152(
        in_channels = 6,
        skip_channels: int,
        out_channels: int,
    )

    # Define loss function
    if config.num_classes == 1:
        # Binary segmentation with BCE+Dice
        criterion = BCEDiceLoss(dice_weight=0.5, bce_weight=0.5)
        print("using dice loss")
    else:
        # For multi-class, we would need to adapt the loss function
        # This is a simple implementation for multi-class that could be improved
        criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Define scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=True
    )

    # Create trainer with visualization
    visualize_every = 10

    # Resume from checkpoint if available
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_metric = 0.0
    global_step = 0

    if config.resume_training:
        # Try to resume from checkpoint if it exists
        try:
            start_epoch, best_val_loss, best_val_metric, global_step = resume_from_checkpoint(
                config, model, optimizer, scheduler, load_best=False
            )
            print(f"Resuming training from epoch {start_epoch}")
        except Exception as e:
            print(f"Starting training from scratch: {e}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        scheduler=scheduler,
        visualize_every=visualize_every
    )

    # Update trainer with checkpoint information
    trainer.best_val_loss = best_val_loss
    trainer.best_val_metric = best_val_metric
    trainer.global_step = global_step

    # Modify trainer's train method to start from the specified epoch
    original_num_epochs = config.num_epochs
    if start_epoch > 0:
        # Adjust the number of epochs to train for, so we still train for the full specified amount
        config.num_epochs = original_num_epochs - start_epoch

    # Train model
    history = trainer.train()

    print("Training completed!")
    print(f"Best validation metric: {trainer.best_val_metric:.4f}")
    print(f"Best model saved at: {os.path.join(config.checkpoint_dir, 'best_model.pth')}")


if __name__ == "__main__":
    main()
