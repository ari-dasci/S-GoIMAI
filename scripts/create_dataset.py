from transformation import *
import argparse
from dataset import Dataset

parser = argparse.ArgumentParser(description= "Python script to prepare a dataset for training and applying data augmentation techniques.")
parser.add_argument("-o", "--output", help="Set the output path to save the dataset.", default="./")
parser.add_argument("-i", "--input", help="Set the input path for the dataset used.", required=True)
parser.add_argument("-n", "--num_images", help="Set the number of images to generate from Data Augmentation.", default=10)
parser.add_argument("-t", "--threads", type=int, help="Set the number of threads that the script will use.", default=1)
parser.add_argument("-v", "--verbose", action="store_true")

args = parser.parse_args()

if args.verbose:
    verbose = True
    print("Verbose")
else:
    verbose = False

if args.input:
    dataset_dir = args.input
    print(f"Dataset path: {dataset_dir}")

if args.output:
    output_dir = args.output
    print(f"Save path: {output_dir}")

if args.num_images:
    num_images = args.num_images
    print(f"Number of images to generate: {num_images}")

if args.threads:
    threads = args.threads
    print(f"Number of threads: {threads}")

augs = Compose(
    [
        RandomRotation(-45.0, 45.0),
        # RandomBrightness(0.8, 1.2),
        # RandomFlip(),
        # RandomBrightness(0.8, 1.2),
        # Resize(224, 224),
    ]
)

Dataset(dataset_dir, output_dir, splits=splits, augmentations=augs, nproc=threads, num_images_class=0.25, seed=1234).data_augmentation(num_images)
