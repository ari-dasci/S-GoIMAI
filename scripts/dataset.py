import os
from PIL import Image
import numpy as np
import multiprocessing
from rich.progress import Progress
import functools
import random
import json


def extract_patches(image, size):

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


class Dataset:

    def __init__(self, input_dir, output_dir=None, nproc=1, augmentations=None, num_images_class = 1.0, seed=None):
        """
        @input_dir: ruta donde se encuentran las imágenes.
        @output_dir: ruta donde se guardan las imágenes tras el aumento de datos.
        @nproc: número de procesos a utilizar para generar las imágenes en paralelo. Por defecto tiene de valor 1, cambiar a -1 si se quieren utilizar todas las hebras.
        @augmentations: funciones que se aplican como aumento de datos al conjunto de train.
        @preprocessing: funciones que se aplican como preprocesamiento a las imágenes.
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

        print("Setup")
        # Keep only self.num_images_per_class
        self.images = {}
        for class_dir in self.splits:
            label = class_dir.split('/')[-1]
            images = list(os.listdir(class_dir))
            num_images = int(self.num_images_per_class * len(images))
            for img in np.random.choice(images, num_images, replace=False):
                img_path = os.path.join(class_dir, img)
                self.images[img_path] = label
        print(f'Number of images {len(self.images)}')

        print("End setup")

        self.metadata = {
            'original_path': self.input_dir,
            'seed': self.seed,
            'augmentations': [str(aug) for aug in self.augmentations.aug_list],
        }

    def data_augmentation(self, n, nproc=-1):

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
