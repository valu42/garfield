import pathlib
from dataclasses import dataclass
import  torch

# Base project directory (two levels up from config.py)
def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

@dataclass
class Config:
    """Configuration class for segmentation model training."""
    # Path
    project_folder = pathlib.Path(__file__).parent.parent.parent.absolute()
    data_dir: pathlib.Path = project_folder / "data"
    checkpoint_dir: pathlib.Path = project_folder / "checkpoints"
    visualization_dir: pathlib.Path = project_folder / "visualizations"
    split_file_train: pathlib.Path = data_dir / "splits" / "train.txt"
    split_file_val: pathlib.Path = data_dir / "splits" / "val.txt"

    # Model
    num_classes: int = 1  # Binary segmentation by default
    pretrained: bool = True

    # Training
    batch_size: int = 64
    num_epochs: int = 500
    lr: float = 1e-4
    override_lr: float = True
    weight_decay: float = 1e-5
    device: str = get_device()
    num_workers: int = 4

    # Checkpointing
    save_best_only: bool = True
    early_stopping_patience: int = 150
    resume_training: bool = False
