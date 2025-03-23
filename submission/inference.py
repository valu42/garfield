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


def get_transform(img_size=None):
    """
    Get image transformation pipeline for standard (non-TTA) inference.
    """
    # ImageNet normalization values
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if img_size:
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
    Convert a binary mask (numpy array) to run-length encoding.
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# At the top of your file, add:
from unet152 import build_unet
from swin import build_swin_large_unet
from swin2 import build_swin2_large_unet  # New import for Swin2

def load_model(model_path, device, model_type='unet', num_classes=1, img_size=256, window_size=8):
    """
    Load a model from a checkpoint given its type.
    """
    if model_type.lower() == 'unet':
        model = build_unet(num_classes=num_classes, pretrained=False)
    elif model_type.lower() == 'swin':
        model = build_swin_large_unet(
            img_size=img_size,
            num_classes=num_classes,
            window_size=window_size,
            pretrained=False
        )
    elif model_type.lower() == 'swin2':  # New branch for Swin2
        model = build_swin2_large_unet(
            img_size=img_size,
            num_classes=num_classes,
            pretrained=True
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose 'unet', 'swin', or 'swin2'.")

    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        # If checkpoint contains a state dict key, use it; otherwise assume checkpoint is the state dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning: Strict loading failed, trying non-strict: {e}")
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

    model = model.to(device)
    model.eval()
    return model



def load_models(model_configs, device):
    """
    Load multiple models for ensemble inference.
    """
    models = []
    for i, config in enumerate(model_configs):
        try:
            model_path = config['path']
            model_type = config.get('type', 'unet')
            num_classes = config.get('num_classes', 1)
            img_size = config.get('img_size', 256)
            window_size = config.get('window_size', 8)

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
            print("Skipping this model.")
    if not models:
        raise ValueError("No models were successfully loaded. Cannot continue.")
    return models


class StandardImageDataset(Dataset):
    """
    Dataset for standard (non-TTA) inference.
    Applies the given transform to each image.
    """
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'path': str(img_path)}


def standard_collate(batch):
    """
    Custom collate function for standard inference.
    """
    images = torch.stack([item['image'] for item in batch])
    paths = [item['path'] for item in batch]
    return {'image': images, 'path': paths}


def predict_batch_ensemble_no_tta(models, images, device, threshold=0.5, use_amp=False,
                                  ensemble_method='mean', weights=None):
    """
    Run inference on a batch of images using the ensemble of models without TTA.
    Aggregates predictions using the specified ensemble method.
    Supports weighting if ensemble_method is 'weighted'.
    """
    if len(models) == 1:
        with torch.no_grad():
            if use_amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = models[0](images.to(device))
            else:
                outputs = models[0](images.to(device))
        probs = torch.sigmoid(outputs)
        preds = (probs > threshold).float()
        return probs, preds

    all_model_probs = []
    with torch.no_grad():
        for model in models:
            if use_amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(images.to(device))
            else:
                outputs = model(images.to(device))
            probs = torch.sigmoid(outputs)
            all_model_probs.append(probs)

    if ensemble_method == 'weighted':
        if weights is None:
            raise ValueError("Weights must be provided for weighted ensemble.")
        if len(weights) != len(models):
            raise ValueError("The number of weights must match the number of models.")
        # Normalize weights and ensure they are on the correct device
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        weights_tensor = weights_tensor / weights_tensor.sum()
        # Reshape weights to match the dimensions of stacked_probs (which is 5D)
        weights_tensor = weights_tensor.view(-1, 1, 1, 1, 1)
        # Stack model probabilities
        stacked_probs = torch.stack(
            all_model_probs)  # Expected shape: (num_models, batch_size, channels, height, width)
        # Debug prints:
        print("Stacked_probs shape:", stacked_probs.shape)
        print("Weights tensor shape after view:", weights_tensor.shape)
        # Compute weighted sum of probabilities
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
        # Hard voting after thresholding
        votes = torch.stack([(prob > threshold).float() for prob in all_model_probs])
        ensemble_preds = (torch.sum(votes, dim=0) > (len(models) / 2)).float()
        # Also compute average probabilities for consistency
        ensemble_probs = torch.mean(torch.stack(all_model_probs), dim=0)
        return ensemble_probs, ensemble_preds
    else:
        # Default to mean aggregation if an unrecognized method is provided
        ensemble_probs = torch.mean(torch.stack(all_model_probs), dim=0)

    ensemble_preds = (ensemble_probs > threshold).float()
    return ensemble_probs, ensemble_preds


def csv_inference_ensemble_no_tta(models, input_dir, csv_path, transform,
                                  batch_size=8, device='cuda', threshold=0.5,
                                  num_workers=4, csv_batch_size=1000, use_amp=False,
                                  ensemble_method='mean', weights=None, img_size=256):
    """
    Process images using the ensemble of models without TTA and save predictions to a CSV file.
    Supports weighted ensemble if weights are provided and ensemble_method is 'weighted'.
    """
    # Get all image files from the input directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(input_dir).glob(f'*{ext}')))
    image_files.sort()

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    # Check if CSV already exists and load processed images
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

    # Create dataset and dataloader for standard inference
    dataset = StandardImageDataset(to_process, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if str(device).startswith('cuda') else False,
        collate_fn=standard_collate
    )

    start_time = time.time()
    processed_count = 0
    all_results = []

    print(f"Using ensemble of {len(models)} models with {ensemble_method} aggregation (non-TTA)")

    if use_amp and device.type == 'cuda':
        print("Using Automatic Mixed Precision (AMP)")
    else:
        print("Automatic Mixed Precision disabled")

    # Process images in batches
    for batch in tqdm(dataloader, desc="Processing images"):
        images = batch['image']
        batch_paths = batch['path']

        # Run inference with ensemble (non-TTA)
        _, preds = predict_batch_ensemble_no_tta(
            models, images, device, threshold, use_amp, ensemble_method, weights
        )

        # Process predictions and generate RLE encodings
        for j, img_path_str in enumerate(batch_paths):
            image_id = Path(img_path_str).stem
            mask = preds[j].squeeze().cpu().numpy()
            rle = mask2rle(mask)
            all_results.append({
                'ImageId': image_id,
                'EncodedPixels': rle
            })

        processed_count += len(batch_paths)

        # Write results to CSV periodically
        if len(all_results) >= csv_batch_size:
            batch_df = pd.DataFrame(all_results)
            updated_df = pd.concat([existing_df, batch_df], ignore_index=True)
            updated_df.to_csv(csv_path, index=False)
            existing_df = updated_df
            all_results = []

        # Print processing speed periodically
        if processed_count % (batch_size * 5) < batch_size:
            elapsed = time.time() - start_time
            images_per_second = processed_count / elapsed
            print(f"Processing speed: {images_per_second:.2f} images/second")

    # Write any remaining results to CSV
    if all_results:
        batch_df = pd.DataFrame(all_results)
        updated_df = pd.concat([existing_df, batch_df], ignore_index=True)
        updated_df.to_csv(csv_path, index=False)

    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average speed: {processed_count / total_time:.2f} images/second")
    print(f"Inference complete. Results saved to {csv_path}")


def main():
    # Model configurations for ensemble inference
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
            'window_size': 7  # Adjust to match the pretrained Swin models if needed
        },
        {
            'path': "best_swin2.pth",  # New configuration for Swin2
            'type': "swin2",
            'num_classes': 1,
            'img_size': 256,
        }
    ]

    input_dir = "../data/test/images"
    csv_path = "./submission_no_tta.csv"  # Save output CSV in the current directory

    # Inference parameters
    batch_size = 8
    threshold = 0.5
    img_size = 256
    num_workers = 4
    csv_batch_size = 1000
    # Options for ensemble_method: 'mean', 'median', 'max', 'min', 'vote', 'weighted'
    ensemble_method = 'weighted'
    # Set weights for weighted ensemble, for example, 30% for the first model and 70% for the second model
    weights = [0.1, 0.45, 0.45]

    # Determine device and AMP usage
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

    # Get the transformation for standard (non-TTA) inference
    transform = get_transform(img_size)

    # Load ensemble models
    models = load_models(model_configs, device)
    print(f"Loaded {len(models)} models for ensemble inference")

    # Run standard (non-TTA) inference with ensemble
    csv_inference_ensemble_no_tta(
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

    print("Inference completed!")


if __name__ == "__main__":
    main()
