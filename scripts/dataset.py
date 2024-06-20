import os
from PIL import Image
import numpy as np
import multiprocessing
from rich.progress import Progress
import functools
import random
import json
import tensorflow as tf
from typing import Tuple


def extract_patches(image, size):
    """
    Split an image into equal-sized patches.
    
    Args:
        image: PIL Image - source image to split
        size: int - pixel dimensions for square patches (size x size)
    
    Returns:
        list: PIL Image objects representing the patches
    """
    w, h = image.size
    ncols = w // size
    nrows = h // size

    patch_width = size
    patch_height = size

    patches = []

    for row in range(nrows):
        for col in range(ncols):
            # Calculate the patch coordinates
            left = col * patch_width
            upper = row * patch_height
            right = left + patch_width
            lower = upper + patch_height

            patch = image.crop((left, upper, right, lower))
            patches.append(patch)

    return patches

def extract_patches_tf(image: tf.Tensor, patch_size: int, target_size: Tuple[int, int]):
    """TensorFlow implementation of non-overlapping patch extraction with resizing"""
    # Convert to batch of 1 for TF ops
    image = tf.expand_dims(image, 0)  
    
    # Extract patches [1, num_patches, patch_size, patch_size, 3]
    patches = tf.image.extract_patches(
        images=image,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    
    # Reshape and resize patches
    patch_dims = patches.shape[1] * patches.shape[2]
    patches = tf.reshape(patches, [patch_dims, patch_size, patch_size, 3])
    patches = tf.image.resize(patches, target_size, method='bicubic')
    
    return patches


class Dataset:
    """
    Main dataset class for handling image augmentation and processing.
    
    Features:
    - Parallel processing of images
    - Class-aware image selection
    - Metadata tracking
    - Deterministic sampling with seed control
    """

    def __init__(self, input_dir, output_dir=None, nproc=1, augmentations=None, num_images_class=1.0, seed=None):
        """
        Initialize dataset processor.
        
        Args:
            input_dir: str - Path to directory with class subdirectories
            output_dir: str - (Optional) Custom output path for augmented images
            nproc: int - Number of parallel processes (-1 = all cores)
            augmentations: Compose - Augmentation transformations pipeline
            num_images_class: float - Fraction of images per class to use (0.0-1.0)
            seed: int - Random seed for reproducible sampling
        """

        self.input_dir = input_dir
        # Check if dir exist
        if not os.path.exists(self.input_dir):
            raise Exception("{} doesn't exists".format(self.input_dir))

        # Create a datasets dir in the parent of input_dir if needed
        # where the new dataset with data augmentation is going to be saved
        if output_dir is None:
            self.output_dir = os.path.dirname(self.input_dir)
            self.output_dir = os.path.join(self.output_dir, "datasets")
        else:
            self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        self.nproc = nproc
        self.augmentations = augmentations
        self.num_images_per_class = num_images_class

        print("Input dir: {}".format(self.input_dir))
        print("Output dir: {}".format(self.output_dir))

        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.setup()

    def setup(self):
        """
        Prepare dataset structure by:
        1. Scanning input directory for class folders
        2. Randomly selecting subset of images per class
        3. Building image-class mapping
        
        Uses num_images_class to determine sample size per class.
        Maintains reproducible randomization through fixed seed.
        """
        print("Setup")
        self.images = {}
        # Process class directories directly from input dir
        for class_name in os.listdir(self.input_dir):
            class_dir = os.path.join(self.input_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            images = list(os.listdir(class_dir))
            num_images = int(self.num_images_per_class * len(images))
            for img in np.random.choice(images, num_images, replace=False):
                img_path = os.path.join(class_dir, img)
                self.images[img_path] = class_name  # Use directory name as class
        print(f'Number of images {len(self.images)}')

        print("End setup")

        self.metadata = {
            'original_path': self.input_dir,
            'seed': self.seed,
            'augmentations': [str(aug) for aug in self.augmentations.aug_list],
        }

    def data_augmentation(self, n, nproc=-1):
        """
        Generate augmented images through parallel processing.
        
        Args:
            n: int - Number of augmented versions to create per image
            nproc: int - Number of parallel workers (-1 = all cores)
            
        Creates:
            - Output directory structure preserving original hierarchy
            - metadata.json with processing parameters and augmentation details
        """

        self.metadata['n_augmentations'] = n

        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as fp:
            json.dump(self.metadata, fp, indent=4)

        dir_name = self.input_dir.split("/")[-1]
        self.output_dir = os.path.join(self.output_dir, "{}-DA-{}".format(dir_name, n))
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        with Progress() as progress:
            if nproc == -1:
                self.pool = multiprocessing.Pool()
            else:
                self.pool = multiprocessing.Pool(nproc)

            task = progress.add_task("[cyan]Data augmentation", total=1)
            data = self.images.keys()

            for result in self.pool.imap_unordered(
                    functools.partial(
                        self.image_processing,
                        path=self.output_dir,
                        augmentations=self.augmentations,
                        n=n
                    ),
                    list(data)):
                progress.update(task, advance=1)

            self.pool.close()
            self.pool.join()



    @staticmethod
    def image_processing(img, path, augmentations, n):
        """
        Process individual images with augmentations.
        
        Args:
            img: str - Path to source image
            path: str - Base output directory
            augmentations: Compose - Transformation pipeline
            n: int - Number of variants to generate
            
        Output Files:
            Saves images in format: {class_path}/{base_name}-{index}.jpg
            Maintains original file naming with augmentation index suffix
        """
        print(img)

        img_name = img.split("/")[-1]
        # label = img.split('/')[-2]
        img_path = os.path.dirname(os.path.dirname(img))

        # Open the image
        im = Image.open(img)
        base_name, ext = img_name.split('.')  # Remove the file extension

        da_path = os.path.join(path, "data_aug-v3-{}".format(n))

        os.makedirs(da_path, exist_ok=True)

        # Create n images from the original
        for i in range(n):
            nimg = augmentations(im)  # Apply all the augmentations
            class_path = os.path.join(da_path, base_name)
            os.makedirs(class_path, exist_ok=True)
            nimg.save(os.path.join(class_path, "{}-{}.jpg".format(base_name, i)), quality=95)
