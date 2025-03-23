# CV Hackathon Submission - tiiminimi Garfield

This repository contains code for training and evaluating deep learning models for segmenting AI-generated content in images. 
The implementation uses PyTorch and architectures like Swin Transformer, Swin Transformer V2, and ResNet152-based UNet.

## Approach
As the data was already quite clean, the only preprocessing we did was splitting the data.

Our final approach uses an ensemble of three different model architectures:

1. **Swin Transformer UNet**: A UNet-like architecture with a Swin Transformer backbone
2. **Swin Transformer V2 UNet**: Similar to the above but using the improved Swin Transformer V2
3. **ResNet152 UNet**: A UNet with ResNet152 encoder

The final prediction is produced by ensembling these three models with test-time augmentations (TTA) including rotations and flips.
The ensembling is simply weighted average of the predictions of the three models (0.4, 0.4, 0.2).

During training we used only random flips, rotations and crop augmentations. As loss we used BCEWithLogitsLoss combined with DiceLoss
and Adam optimizer. We started with learning rate 1e-4 and used ReduceLROnPlateau scheduler to reduce the learning rate by a factor of 0.5
if the validation loss did not improve for 5 epochs.

We picked the models based on the validation dice coefficient. 

## Installation

### Requirements

- Python >= 3.12
- torch >= 2.6.0
- torchvision >= 0.21.0
- numpy >= 2.2.4
- scikit-learn >= 1.6.1
- albumentations >= 2.0.5
- tqdm >= 4.67.1
- matplotlib >= 3.10.1
- einops >= 0.8.1
- timm >= 0.6.0 (for Swin Transformer models)

This project uses Poetry for dependency management. To install and set up the project just use poetry install


# Training guide

### Data Preparation

1. **Organize your dataset** in the following structure:
data/
├── train/
│   ├── images/        # Contains image_{id}.png files
│   └── masks/         # Contains corresponding image_{id}.png files with the same names

2. **Run the data splitting script** with the following command: python -m src.data.data_preparation.create_split

3. **Pick the model you want to train**: Add some of these lines to the train.py where there'sc urrently the build_swin2_large_unet function call:

model = build_unet(num_classes=1, pretrained=True)                   # ResNet152 UNet
model = build_swin_large_unet(img_size=256, num_classes=1, pretrained=True)  # Swin Transformer
model = build_swin2_large_unet(img_size=256, num_classes=1, pretrained=True) # Swin Transformer V2

4. **Run the training script with the following command**: poetry run python -m src.training.train



