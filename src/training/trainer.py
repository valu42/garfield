import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Optional, Callable, Union, Tuple
from torch.cuda.amp import autocast, GradScaler

from src.configs.config import Config
from src.utils.visualize import visualize_batch

import matplotlib.pyplot as plt


class Trainer:
    """
    Trainer class for segmentation model training and validation.
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module,
            optimizer: optim.Optimizer,
            config: Config,
            scheduler: Optional[object] = None,
            visualize_every: int = 0  # 0 means no visualization
    ):
        """
        Initialize the trainer.

        Args:
            model: The model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function
            optimizer: Optimizer
            config: Configuration object
            scheduler: Learning rate scheduler (optional)
            visualize_every: Visualize training every N batches (0 to disable)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.visualize_every = visualize_every

        # Create checkpoint directory if it doesn't exist
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        # Create visualization directory if needed
        if visualize_every > 0:
            os.makedirs(self.config.visualization_dir, exist_ok=True)

        # Move model to device
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Setup AMP (Automatic Mixed Precision)
        self.use_amp = torch.cuda.is_available() and 'cuda' in config.device
        self.scaler = GradScaler() if self.use_amp else None

        if self.use_amp:
            print("Using Automatic Mixed Precision (AMP) for training")

        # Initialize tracking variables
        self.best_val_loss = float('inf')
        self.best_val_metric = 0.0
        self.epochs_without_improvement = 0
        self.global_step = 0

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []

        # Store validation batch for visualization
        self.val_vis_batch = None
        if visualize_every > 0:
            self._cache_validation_batch()

    def _cache_validation_batch(self):
        """Cache a validation batch for visualization during training."""
        try:
            self.val_vis_batch = next(iter(self.val_loader))
        except:
            print("Warning: Could not cache validation batch for visualization")
            self.val_vis_batch = None

    def train(self) -> Dict[str, List[float]]:
        """
        Train the model for the specified number of epochs.

        Returns:
            Dictionary with training history
        """
        print(f"Starting training for {self.config.num_epochs} epochs")
        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}/{self.config.num_epochs} - Learning Rate: {current_lr:.6f}")

            # Train for one epoch
            train_loss = self.train_one_epoch(epoch)
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_metric = self.validate()
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metric)

            # Print epoch results
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Metric: {val_metric:.4f}")

            # Update scheduler if provided
            if self.scheduler is not None:
                self.scheduler.step(val_metric)  # For ReduceLROnPlateau

            # Check if this is the best model
            is_best = val_metric > self.best_val_metric

            if is_best:
                self.best_val_metric = val_metric
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                # Save checkpoint
                self._save_checkpoint(epoch, is_best)
            else:
                self.epochs_without_improvement += 1
                # Save checkpoint if not saving best only
                if not self.config.save_best_only:
                    self._save_checkpoint(epoch, is_best)

            # Early stopping
            if self.config.early_stopping_patience > 0 and self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'val_metric': self.val_metrics
        }

    def train_one_epoch(self, epoch: int) -> float:
        """
        Train the model for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            # Increment global step counter
            self.global_step += 1

            # Get data and move to device
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device).unsqueeze(1)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass with AMP if available
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                # Backward pass and optimize with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward and backward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()

            # Update running loss
            epoch_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())

            # Visualize training if needed
            if self.visualize_every > 0 and (self.global_step - 1) % self.visualize_every == 0:
                if epoch == 0:
                    print(f"Loss: {loss.item()}")
                self._visualize_training(batch, epoch, batch_idx)

        # Calculate average loss
        avg_loss = epoch_loss / len(self.train_loader)
        return avg_loss

    def _visualize_training(self, train_batch, epoch, batch_idx):
        """Generate and save visualizations of the current training state."""

        # Visualize training batch
        visualize_batch(
            model=self.model,
            batch=train_batch,
            device=self.device,
            output_dir=self.config.visualization_dir,
            epoch=epoch,
            batch_idx=batch_idx,
            max_samples=4,
            save_fig=True
        )

        # Close all plots to avoid memory leaks
        plt.close('all')

    def validate(self) -> Tuple[float, float]:
        """
        Validate the model on the validation set.

        Returns:
            Tuple of (validation loss, validation metric)
        """
        self.model.eval()
        val_loss = 0.0
        val_metric = 0.0

        progress_bar = tqdm(self.val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                # Get data and move to device
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device).unsqueeze(1)

                # Forward pass with AMP if available
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                # Update running loss
                val_loss += loss.item()

                # Calculate metric (e.g., Dice coefficient)
                batch_metric = self._compute_dice_coefficient(outputs, masks)
                val_metric += batch_metric

                # Update progress bar
                progress_bar.set_postfix(loss=loss.item(), dice=batch_metric)

        # Calculate average loss and metric
        avg_loss = val_loss / len(self.val_loader)
        avg_metric = val_metric / len(self.val_loader)

        return avg_loss, avg_metric

    def _save_checkpoint(self, epoch: int, is_best: bool) -> None:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': self.best_val_loss,
            'val_metric': self.best_val_metric,
            'global_step': self.global_step,
        }

        # Save scheduler state if it exists
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save scaler state if using AMP
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Save latest checkpoint
        checkpoint_path = os.path.join(self.config.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model with validation metric: {self.best_val_metric:.4f}")

    def _compute_dice_coefficient(self, outputs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-7) -> float:
        """
        Compute Dice coefficient.

        Args:
            outputs: Model predictions (logits)
            targets: Ground truth masks
            smooth: Smoothing factor to avoid division by zero

        Returns:
            Dice coefficient value
        """
        # Apply sigmoid to outputs
        probs = torch.sigmoid(outputs)

        # Convert probabilities to binary predictions
        preds = (probs > 0.5).float()

        # Flatten predictions and targets
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

        # Calculate intersection and union
        intersection = (preds_flat * targets_flat).sum()
        union = preds_flat.sum() + targets_flat.sum()

        # Calculate Dice coefficient
        dice = (2.0 * intersection + smooth) / (union + smooth)

        return dice.item()