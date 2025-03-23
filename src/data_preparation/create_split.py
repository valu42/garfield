import os
import random
import glob


def split_train_val(folder_path, output_path, train_ratio=0.8):
    """
    Split image files in a folder into training and validation sets.

    Args:
        folder_path (str): Path to the folder containing the images
        output_path (str): Path where train.txt and val.txt will be saved
        train_ratio (float): Ratio of images to use for training (default: 0.8)

    Returns:
        None (writes to train.txt and val.txt files in the specified output location)
    """
    # Get all image files matching the pattern
    image_files = glob.glob(os.path.join(folder_path, "image_*.png"))

    print(f"Found {len(image_files)} images matching the pattern.")

    # Extract image IDs from filenames
    image_ids = []
    for file_path in image_files:
        filename = os.path.basename(file_path)
        # Extract the ID part between "image_" and ".png"
        image_id = filename.replace("image_", "").replace(".png", "")
        image_ids.append(image_id)

    # Shuffle the image IDs randomly
    random.shuffle(image_ids)

    # Calculate the split index
    split_idx = int(len(image_ids) * train_ratio)

    # Split into training and validation sets
    train_ids = image_ids[:split_idx]
    val_ids = image_ids[split_idx:]

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Write to train.txt in the specified output location
    train_path = os.path.join(output_path, "train.txt")
    with open(train_path, "w") as f:
        for image_id in train_ids:
            f.write(f"image_{image_id}.png\n")

    # Write to val.txt in the specified output location
    val_path = os.path.join(output_path, "val.txt")
    with open(val_path, "w") as f:
        for image_id in val_ids:
            f.write(f"image_{image_id}.png\n")

    print(f"Split complete: {len(train_ids)} images in train set, {len(val_ids)} images in validation set")
    print(f"Train file saved to: {os.path.abspath(train_path)}")
    print(f"Validation file saved to: {os.path.abspath(val_path)}")


# Example usage
if __name__ == "__main__":
    # Path to folder containing the images
    folder_path = "data/train/images"
    # Path where train.txt and val.txt will be saved
    output_path = "data/splits"  # Change this to your desired output location

    ratio = 0.8

    split_train_val(folder_path, output_path, ratio)

