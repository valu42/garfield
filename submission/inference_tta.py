import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import time
import torch.nn.functional as F

# Import model architectures
from unet152 import build_unet
from swin import build_swin_large_unet
from swin2 import build_swin2_large_unet  # New import for Swin2

def get_transform(img_size=None, augment=False):
    """
    Get image transformation pipeline for inference.
    When augment=True, return only normalization for use with TTA functions.
    """
    # ImageNet normalization values
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if augment:
        # For TTA, we'll handle resize separately
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    elif img_size:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def load_model(model_path, device, model_type='unet', num_classes=1, img_size=256, window_size=8):
    """Load the model from checkpoint based on model type."""
    # Create model based on type
    if model_type.lower() == 'unet':
        model = build_unet(num_classes=num_classes, pretrained=False)
    elif model_type.lower() == 'swin':
        model = build_swin_large_unet(
            img_size=img_size,
            num_classes=num_classes,
            window_size=window_size,
            pretrained=False
        )
    elif model_type.lower() == 'swin2':  # New branch for Swin2 support
        model = build_swin2_large_unet(
            img_size=img_size,
            num_classes=num_classes,
            pretrained=True
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose 'unet', 'swin', or 'swin2'.")

    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location='cpu')

        # Check if model_state_dict exists in the checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # Try to use the checkpoint directly as a state dict
            state_dict = checkpoint

        # Handle strict parameter loading
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning: Strict loading failed, attempting non-strict loading: {e}")
            model.load_state_dict(state_dict, strict=False)

        print(f"Model loaded from {model_path}")
        print(f"Model type: {model_type}")
        print("Checkpoint information:")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Validation Loss: {checkpoint.get('val_loss', 'N/A')}")
        print(f"  Validation Metric: {checkpoint.get('val_metric', 'N/A')}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()

    return model

def load_models(model_configs, device):
    """
    Load multiple models for ensemble.

    model_configs: list of dicts with keys 'path', 'type', and optional parameters
    Example: [
        {'path': 'model1.pth', 'type': 'unet', 'num_classes': 1},
        {'path': 'model2.pth', 'type': 'swin', 'num_classes': 1, 'img_size': 256, 'window_size': 7},
        {'path': 'model3.pth', 'type': 'swin2', 'num_classes': 1, 'img_size': 256}
    ]
    """
    models = []
    for i, config in enumerate(model_configs):
        try:
            model_path = config['path']
            model_type = config.get('type', 'unet')  # Default to unet if not specified
            num_classes = config.get('num_classes', 1)  # Default to 1 if not specified
            img_size = config.get('img_size', 256)  # Default to 256 if not specified
            window_size = config.get('window_size', 8)  # Default to 8 if not specified

            print(f"Loading model {i + 1}/{len(model_configs)}: {model_path} (type: {model_type})")

            model = load_model(
                model_path=model_path,
                device=device,
                model_type=model_type,
                num_classes=num_classes,
                img_size=img_size,
                window_size=window_size
            )
            models.append(model)
            print(f"Successfully loaded model {i + 1}")
        except Exception as e:
            print(f"Error loading model {i + 1}: {e}")
            print("Skipping this model and continuing with others")

    if not models:
        raise ValueError("No models were successfully loaded. Cannot continue.")

    return models

# Custom dataset with support for TTA
class TTAImageDataset(Dataset):
    def __init__(self, image_files, transform=None, img_size=None, tta=False):
        self.image_files = image_files
        self.transform = transform
        self.img_size = img_size
        self.tta = tta

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        # Load image
        image = Image.open(img_path).convert('RGB')

        # If using TTA, store original image for augmentations later
        original_image = image

        # Apply transforms if any
        if self.transform:
            if self.tta:
                # For TTA, handle resize separately to avoid multiple resizes for different augmentations
                if self.img_size:
                    original_image = original_image.resize((self.img_size, self.img_size))
                # Don't transform here; we'll do it during TTA
                image = original_image
            else:
                # Regular non-TTA pipeline
                image = self.transform(image)

        return {
            'image': image,
            'path': str(img_path),  # Convert Path to string to avoid collate issues
            'original_image': original_image if self.tta else None
        }

# Custom collate function to handle paths and original images for TTA
def tta_collate(batch):
    if batch[0]['original_image'] is not None:
        # TTA mode - return original images and paths
        original_images = [item['original_image'] for item in batch]
        paths = [item['path'] for item in batch]
        return {'original_image': original_images, 'path': paths}
    else:
        # Regular mode - return transformed images and paths
        images = torch.stack([item['image'] for item in batch])
        paths = [item['path'] for item in batch]
        return {'image': images, 'path': paths}

# TTA functions
def apply_tta_augmentations(images, transform, device):
    """Apply Test Time Augmentation to a batch of PIL images."""
    augmentations = []

    # Original
    orig_batch = torch.stack([transform(img) for img in images]).to(device)
    augmentations.append(orig_batch)

    # Horizontal flip
    hflip_batch = torch.stack([transform(img.transpose(Image.FLIP_LEFT_RIGHT)) for img in images]).to(device)
    augmentations.append(hflip_batch)

    # Vertical flip
    vflip_batch = torch.stack([transform(img.transpose(Image.FLIP_TOP_BOTTOM)) for img in images]).to(device)
    augmentations.append(vflip_batch)

    # 90 degree rotation
    rot90_batch = torch.stack([transform(img.transpose(Image.ROTATE_90)) for img in images]).to(device)
    augmentations.append(rot90_batch)

    # 180 degree rotation
    rot180_batch = torch.stack([transform(img.transpose(Image.ROTATE_180)) for img in images]).to(device)
    augmentations.append(rot180_batch)

    # 270 degree rotation
    rot270_batch = torch.stack([transform(img.transpose(Image.ROTATE_270)) for img in images]).to(device)
    augmentations.append(rot270_batch)

    return augmentations

def reverse_tta_augmentations(aug_predictions):
    """Reverse the TTA augmentations to align predictions with original orientation."""
    orig_pred = aug_predictions[0]

    # Horizontal flip
    hflip_pred = torch.flip(aug_predictions[1], dims=[3])  # Flip horizontally

    # Vertical flip
    vflip_pred = torch.flip(aug_predictions[2], dims=[2])  # Flip vertically

    # 90 degree rotation (rotate 270 to reverse)
    rot90_pred = torch.rot90(aug_predictions[3], k=3, dims=[2, 3])

    # 180 degree rotation (rotate 180 to reverse)
    rot180_pred = torch.rot90(aug_predictions[4], k=2, dims=[2, 3])

    # 270 degree rotation (rotate 90 to reverse)
    rot270_pred = torch.rot90(aug_predictions[5], k=1, dims=[2, 3])

    # Stack all the reversed predictions
    all_preds = torch.stack([orig_pred, hflip_pred, vflip_pred, rot90_pred, rot180_pred, rot270_pred])

    return all_preds

def predict_batch_with_tta(model, original_images, transform, device, threshold=0.5, use_amp=False):
    """Run inference with Test Time Augmentation."""
    # Apply different augmentations
    aug_batches = apply_tta_augmentations(original_images, transform, device)
    aug_predictions = []

    # Get prediction for each augmentation
    with torch.no_grad():
        for aug_batch in aug_batches:
            if use_amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(aug_batch)
            else:
                outputs = model(aug_batch)

            # Apply sigmoid for binary segmentation
            probs = torch.sigmoid(outputs)
            aug_predictions.append(probs)

    # Reverse augmentations to align all predictions
    aligned_predictions = reverse_tta_augmentations(aug_predictions)

    # Average the predictions from all augmentations
    mean_pred = torch.mean(aligned_predictions, dim=0)

    # Apply threshold to get binary masks
    binary_pred = (mean_pred > threshold).float()

    return mean_pred, binary_pred

def predict_batch_ensemble_with_tta(models, original_images, transform, device, threshold=0.5, use_amp=False,
                                    ensemble_method='mean', weights=None):
    """Run ensemble inference with Test Time Augmentation."""
    # If only one model is provided, use single model prediction with TTA
    if len(models) == 1:
        return predict_batch_with_tta(models[0], original_images, transform, device, threshold, use_amp)

    all_model_probs = []

    # Get TTA predictions from each model
    for model in models:
        probs, _ = predict_batch_with_tta(model, original_images, transform, device, threshold, use_amp)
        all_model_probs.append(probs)

    # Combine predictions using the specified method
    if ensemble_method == 'weighted':
        if weights is None:
            raise ValueError("Weights must be provided for weighted ensemble.")
        if len(weights) != len(models):
            raise ValueError("The number of weights must match the number of models.")
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        weights_tensor = weights_tensor / weights_tensor.sum()
        stacked_probs = torch.stack(all_model_probs)  # shape: (num_models, batch_size, channels, H, W)
        weights_tensor = weights_tensor.view(-1, 1, 1, 1, 1)
        ensemble_probs = torch.sum(stacked_probs * weights_tensor, dim=0)
    elif ensemble_method == 'mean':
        ensemble_probs = torch.mean(torch.stack(all_model_probs), dim=0)
    elif ensemble_method == 'median':
        ensemble_probs = torch.median(torch.stack(all_model_probs), dim=0)[0]
    elif ensemble_method == 'max':
        ensemble_probs = torch.max(torch.stack(all_model_probs), dim=0)[0]
    elif ensemble_method == 'min':
        ensemble_probs = torch.min(torch.stack(all_model_probs), dim=0)[0]
    elif ensemble_method == 'vote':
        votes = torch.stack([(prob > threshold).float() for prob in all_model_probs])
        ensemble_preds = (torch.sum(votes, dim=0) > (len(models) / 2)).float()
        ensemble_probs = torch.mean(torch.stack(all_model_probs), dim=0)
        return ensemble_probs, ensemble_preds
    else:
        # Default to mean
        ensemble_probs = torch.mean(torch.stack(all_model_probs), dim=0)

    ensemble_preds = (ensemble_probs > threshold).float()

    return ensemble_probs, ensemble_preds

def tta_csv_inference_ensemble(models, input_dir, csv_path, transform,
                               batch_size=8, device='cuda', threshold=0.5,
                               num_workers=4, csv_batch_size=1000, use_amp=False,
                               ensemble_method='mean', weights=None, img_size=256):
    """
    Process all images using an ensemble of models with Test Time Augmentation.
    Supports weighted ensemble if ensemble_method is 'weighted' and weights are provided.
    """
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(input_dir).glob(f'*{ext}')))

    image_files.sort()

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    # Check if CSV already exists and load it
    processed_images = set()
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            processed_images = set(existing_df['ImageId'].values)
            print(f"Found existing CSV with {len(processed_images)} processed images")
        except Exception as e:
            print(f"Error reading existing CSV: {e}")
            print("Starting from scratch")
            existing_df = pd.DataFrame(columns=['ImageId', 'EncodedPixels'])
    else:
        existing_df = pd.DataFrame(columns=['ImageId', 'EncodedPixels'])

    # Filter out already processed images
    to_process = []
    for img_path in image_files:
        image_id = Path(img_path.name).stem
        if image_id not in processed_images:
            to_process.append(img_path)

    print(f"Found {len(image_files)} total images, {len(to_process)} remaining to process")

    if not to_process:
        print("All images have already been processed!")
        return

    # Create dataset and dataloader
    transform_no_resize = get_transform(augment=True)  # For TTA, we'll resize in each augmentation
    dataset = TTAImageDataset(to_process, transform=None, img_size=img_size, tta=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if str(device).startswith('cuda') else False,
        collate_fn=tta_collate
    )

    start_time = time.time()
    processed_count = 0
    all_results = []

    if len(models) > 1:
        print(f"Using ensemble of {len(models)} models with {ensemble_method} aggregation + TTA")
    else:
        print("Using single model inference with TTA")

    if use_amp and device.type == 'cuda':
        print("Using Automatic Mixed Precision (AMP)")
    else:
        print("Automatic Mixed Precision disabled")

    for batch in tqdm(dataloader, desc="Processing images with TTA"):
        original_images = batch['original_image']
        batch_paths = batch['path']

        # Run inference with TTA and ensemble; pass weights if provided
        _, preds = predict_batch_ensemble_with_tta(
            models, original_images, transform_no_resize, device, threshold, use_amp, ensemble_method, weights
        )

        for j, img_path_str in enumerate(batch_paths):
            image_id = Path(img_path_str).stem
            mask = preds[j].squeeze().cpu().numpy()
            rle = mask2rle(mask)
            all_results.append({
                'ImageId': image_id,
                'EncodedPixels': rle
            })

        processed_count += len(batch_paths)

        if len(all_results) >= csv_batch_size:
            batch_df = pd.DataFrame(all_results)
            updated_df = pd.concat([existing_df, batch_df], ignore_index=True)
            updated_df.to_csv(csv_path, index=False)
            existing_df = updated_df
            all_results = []

        if processed_count % (batch_size * 5) < batch_size:
            elapsed = time.time() - start_time
            images_per_second = processed_count / elapsed
            print(f"Processing speed: {images_per_second:.2f} images/second")

    if all_results:
        batch_df = pd.DataFrame(all_results)
        updated_df = pd.concat([existing_df, batch_df], ignore_index=True)
        updated_df.to_csv(csv_path, index=False)

    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average speed: {processed_count / total_time:.2f} images/second")
    print(f"Completed processing all images. Results saved to {csv_path}")

def main():
    # Model configurations for ensemble
    model_configs = [
        {
            'path': "best_model.pth",
            'type': "unet",
            'num_classes': 1
        },
        {
            'path': "best_swin.pth",
            'type': "swin",
            'num_classes': 1,
            'img_size': 256,
            'window_size': 7  # Set to 7 to match the common pretrained Swin models
        },
        # Uncomment and configure below to add Swin2 support:
        {
            'path': "best_swin2.pth",
            'type': "swin2",
            'num_classes': 1,
            'img_size': 256
        }
    ]

    input_dir = "../data/test/images"
    csv_path = "./submission_tta.csv"

    # Parameters
    batch_size = 8  # Smaller batch size for TTA
    threshold = 0.5
    img_size = 256
    num_workers = 4
    csv_batch_size = 1000
    # To use weighting, set ensemble_method to 'weighted' and provide weights
    ensemble_method = 'weighted'
    weights = [0.1, 0.45, 0.45]  # Example: 20% weight for the first model, 80% for the second
    use_tta = True  # Set to False to disable TTA

    # Determine device and AMP availability
    if torch.cuda.is_available():
        device_name = "cuda"
        use_amp = True
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device_name = "mps"
        use_amp = False
        print("Using Apple Silicon (MPS)")
    else:
        device_name = "cpu"
        use_amp = False
        print("CUDA is not available. Using CPU.")

    device = torch.device(device_name)
    print(f"Using device: {device}")
    print(f"Using {num_workers} CPU workers for data loading")

    # Get transform for non-TTA inference (not used in this TTA script)
    transform = get_transform(img_size)

    # Load models
    models = load_models(model_configs, device)
    print(f"Loaded {len(models)} models for ensemble")

    # Run inference with TTA
    if use_tta:
        print("Running inference with Test-Time Augmentation...")
        tta_csv_inference_ensemble(
            models=models,
            input_dir=input_dir,
            csv_path=csv_path,
            transform=transform,
            batch_size=batch_size,
            device=device,
            threshold=threshold,
            num_workers=num_workers,
            csv_batch_size=csv_batch_size,
            use_amp=use_amp,
            ensemble_method=ensemble_method,
            weights=weights,
            img_size=img_size
        )
    else:
        print("Running standard inference without TTA...")
        raise NotImplementedError("Please enable use_tta or implement a non-TTA inference call")

    print("Inference completed!")

if __name__ == "__main__":
    main()
