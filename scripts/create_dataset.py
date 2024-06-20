from transformation import *
import argparse
from dataset import Dataset

# Main script for generating an augmented dataset from input images
# Usage: python create_dataset.py -i [INPUT_DIR] -o [OUTPUT_DIR] -n [NUM_IMAGES]

# Initialize argument parser with dataset configuration parameters
parser = argparse.ArgumentParser(description="Python script to prepare a dataset for training and applying data augmentation techniques.")
parser.add_argument("-o", "--output", help="Set the output path to save the augmented dataset.", default="./")
parser.add_argument("-i", "--input", help="Path to the original dataset directory (required).", required=True)
parser.add_argument("-n", "--num_images", type=int, help="Number of augmented images to generate per original image.", default=10)
parser.add_argument("-t", "--threads", type=int, help="Number of CPU threads for parallel processing (use -1 for all cores).", default=1)
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output for debugging.")

args = parser.parse_args()

if args.verbose:
    verbose = True
    print("Verbose")
else:
    verbose = False

if args.input:
    dataset_dir = args.input
    print(f"Input dataset directory: {dataset_dir}")

if args.output:
    output_dir = args.output
    print(f"Output directory for augmented dataset: {output_dir}")

if args.num_images:
    num_images = args.num_images
    print(f"Generating {num_images} augmented versions per image")

if args.threads:
    threads = args.threads
    print(f"Using {threads} processing threads")

# Define data augmentation pipeline using Composition of transforms
# Note: Currently only using RandomRotation - uncomment/add other transforms as needed
augs = Compose(
    [
        RandomRotation(-45.0, 45.0),  # Apply random rotation between -45 and +45 degrees
        # Available transforms that can be uncommented:
        # RandomBrightness(0.8, 1.2),  # Adjust brightness randomly between 80-120%
        # RandomFlip(),                # Random horizontal/vertical flips
        # Resize(224, 224),            # Resize images to target dimensions
    ]
)

# Initialize and run dataset augmentation process
# Parameters:
# - num_images_class: Use 25% of original images per class (0.25)
# - seed: Fixed random seed for reproducibility
Dataset(
    dataset_dir,
    output_dir,
    augmentations=augs,
    nproc=threads,
    num_images_class=0.25,  # Use 25% of original images per class
    seed=1234               # Fixed seed for deterministic results
).data_augmentation(num_images)  # Generate augmented images
