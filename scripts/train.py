"""
TensorFlow training script for image classification with EfficientNet models.
Supports k-fold cross-validation and data augmentation.

Features:
- Multiple model architectures (EfficientNetV2 B0-B3)
- Cosine decay learning rate schedule
- Metadata tracking
- Stratified k-fold validation
"""

from pandas._config import dates
import pandas as pd
from PIL import Image
import numpy as np
import pathlib
import os
import argparse
import logging
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
import json
from dataset import extract_patches_tf
import tensorflow as tf
import tensorflow_addons as tfa


msg = "Adding description"

# Initialize parser
parser = argparse.ArgumentParser(description="Train image classification models")
parser.add_argument("-o", "--output", 
                   help="Output directory for saved models", 
                   default="./models")
parser.add_argument("-e", "--epochs", 
                   type=int, 
                   help="Number of training epochs", 
                   default=100)
parser.add_argument("-i", "--input", 
                   help="Path to training data directory",
                   required=True)
parser.add_argument("--original", 
                   help="Path to original unaugmented dataset",
                   required=True)
parser.add_argument("-b", "--batch",
                   help="Batch size for training",
                   type=int, 
                   default=16)
parser.add_argument("-w", "--weights", 
                   help="Model weights initialization",
                   choices=["imagenet", "none"], 
                   default="imagenet")
parser.add_argument("-s", "--size", 
                   help="Input dimensions as WIDTH HEIGHT", 
                   type=int, 
                   nargs=2, 
                   required=True)
parser.add_argument("-m", "--model", 
                   help="Model architecture selection",
                   choices=['b3', 'b2', 'b1', 'b0', 'ir2'], 
                   required=True)
parser.add_argument("-k", "--k_fold", 
                   help="Number of cross-validation folds",
                   type=int,
                   default=5)
parser.add_argument("--loss", 
                   help="Loss function for training",
                   choices=["sparse_categorical_crossentropy"],
                   default="sparse_categorical_crossentropy")

args = parser.parse_args()

if args.input:
    dataset_dir = args.input
    print(f"Augmented dataset path: {args.input}")
    
if args.original:
    original_dataset_dir = args.original
    print(f"Original dataset path: {args.original}")
    
if args.size:
    img_width, img_height = args.size
    print(f"Input size: {img_width}x{img_height}")

if args.loss:
    loss = args.loss
    print(f"Loss function: {loss}")

if args.epochs:
    epochs = args.epochs
    print(f"Training epochs: {epochs}")

if args.batch:
    batch_size = args.batch
    print(f"Batch size: {batch_size}")

if args.weights:
    weights = args.weights
    print(f"Weights: {weights}")

if args.output:

    model_dir = args.output
    print(f"Output dir: {model_dir}")

if args.k_fold:
    k_fold = args.k_fold

models = {
    'b3': tf.keras.applications.EfficientNetV2B3,
    'b2': tf.keras.applications.EfficientNetV2B2,
    'b1': tf.keras.applications.EfficientNetV2B1,
    'b0': tf.keras.applications.EfficientNetV2B0,
    'ir2': tf.keras.applications.inception_resnet_v2.InceptionResNetV2,
}

if args.model:

    model = models[args.model]
    model_name = model
    print(f"Model: {model().name}")


name = datetime.now()
name = name.strftime("%Y-%m-%d_%H-%M-%S")
name = f"{name}-{model_name}-{loss}-{epochs}epochs"
os.makedirs(model_dir, exist_ok=True)
model_dir = os.path.join(model_dir, name)
os.makedirs(model_dir, exist_ok=True)
print(f"Model dir: {model_dir}")

# Create the model metadata
metadata = {
    "name": name,
    "path": model_dir,
    "size" : {'width': img_width, 'height': img_height},
    'batch_size': batch_size,
    'epochs': epochs,
    'weight': weights,
    'model': str(model().name),
    'original_dataset': args.original,
    'augmented_dataset': args.input,
}

with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
    json.dump(metadata, f)


images = list(os.listdir(dataset_dir))
codes = []
for image in images:
    code = image.split('-')[0]

    if code not in codes:
        codes.append(code)
codes = sorted(codes)

num_classes = len(codes)
n_channels = 3
img_size = (img_width, img_height, n_channels)


def get_label(file_path):
    """Extracts class label from file path.
    
    Args:
        file_path: Tensor string with image path
        
    Returns:
        int: Class index from codes list
        
    Note: Expected filename format: "classcode-rest-of-filename.ext"
    """
    parts = tf.strings.split(file_path, os.path.sep)
    filename = tf.strings.split(parts[-1], ".")[0]  # Remove extension
    class_str = tf.strings.split(filename, "-")[0]   # Get class code from filename
    return tf.argmax(class_str == codes)  # Find index in codes list

def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=n_channels)
    img = tf.image.resize(
        img,
        (img_width, img_height)
    )

    return img


def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0,1]
    return img, label

AUTOTUNE = tf.data.AUTOTUNE
def configure_for_performance(ds: tf.data.Dataset):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

def evaluate_with_patches(model, test_ds, codes):
    """Evaluate TF Dataset using patch-based majority voting"""
    total = 0
    correct = 0
    
    for img, label in test_ds.unbatch():
        # Extract and process patches
        patches = extract_patches_tf(img, patch_size=500, target_size=(img_width, img_height))
        
        # Predict and vote
        preds = model.predict(patches, verbose=0)
        predicted_classes = np.argmax(preds, axis=1)
        class_counts = np.bincount(predicted_classes, minlength=len(codes))
        majority = np.argmax(class_counts)
        
        if majority == label:
            correct += 1
        total += 1
    
    return correct / total



def train(train_ds, val_ds=None, test_ds=None, name=None, model=model):
    """Main training procedure.
    
    Args:
        train_ds: Training dataset (tf.data.Dataset)
        val_ds: Optional validation dataset (tf.data.Dataset) - used during training
        test_ds: Optional test dataset (tf.data.Dataset) - used for final evaluation
        name: Base name for saving artifacts
        model: Model constructor function
        
    Builds EfficientNet model with:
    - Customizable input shape
    - Imagenet weights initialization
    - Cosine decay learning rate
    - Classification head for 37 classes
    """
    
    base_model = model(
        include_top=False,
        weights=weights,
        input_shape=img_size)


    # Add the model name to the name used to save everything
    print(f"Base model: {base_model.name}")
    name = f'{name}-{base_model.name}'

    base_model.trainable = True

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x2 = tf.keras.layers.Dense(200, activation="relu")(x)
    output_layer = tf.keras.layers.Dense(37, activation="softmax")(x2)
    model = tf.keras.Model(base_model.input, output_layer)

    decay_steps = int(epochs * (train_ds.cardinality().numpy()))  # Use correct dataset variable
    print('Decay steps:', decay_steps)
    initial_learning_rate = 0.002
    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate, decay_steps)

    adam = tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn)
    lr_metric = get_lr_metric(adam)



    model.compile(
        optimizer=adam, loss=args.loss, metrics=["accuracy", lr_metric],  # Use provided loss argument
    )

    # Training with optional validation
    if not val_ds and not test_ds:
        history = model.fit(train_ds, epochs=epochs, batch_size=batch_size)
    else:
        history = model.fit(train_ds, 
                          validation_data=val_ds,  # Only use validation here
                          epochs=epochs, 
                          batch_size=batch_size)

    # Final evaluation on test set if provided
    if test_ds:
        # Evaluate using patch-based majority voting
        patch_accuracy = evaluate_with_patches(model, test_ds, codes)
        
        name = f'{name}-test_{patch_accuracy:.2f}'
        print(f"\nEvaluation Results:")
        print(f"- Patch Voting Accuracy: {patch_accuracy:.2%}\n")
        
        # Update metadata
        metadata['patch_voting_accuracy'] = patch_accuracy

    try:
        if name is None:
            name = ""
        model.save(os.path.join(model_dir, f"{name}.keras"))

    except Exception as e:
        print(e)
        print("No se ha podido guardar el modelo")

    try:
        print(history)
        pd.DataFrame(history.history).to_csv(os.path.join(model_dir, f"{name}-history.csv"))
    except:
        print("No se ha podido guardar el historial")


train_dir = dataset_dir

# Load a list of images to split
images_list = np.asarray(list(os.listdir(dataset_dir)))
labels_list = np.asarray(list(map(lambda x: x.split('-')[0], images_list)))
# Validate dataset paths
if not os.path.exists(args.original):
    raise FileNotFoundError(f"Original dataset directory not found: {original_dataset_dir}")
if not os.path.exists(args.input):
    raise FileNotFoundError(f"Augmented dataset directory not found: {dataset_dir}")


print(f"\nOriginal dataset: {original_dataset_dir}")
print(f"Augmented dataset: {dataset_dir}\n")

original_images = [os.path.join(original_dataset_dir, classes, img) 
                  for classes in os.listdir(original_dataset_dir) 
                  for img in os.listdir(os.path.join(original_dataset_dir, classes))]

skf = StratifiedKFold(n_splits=k_fold)
for i, (train_index, test_index) in enumerate(skf.split(images_list, labels_list)):
    train_images = [os.path.join(dataset_dir, img, i) for img in images_list[train_index] for i in os.listdir(os.path.join(dataset_dir, img))]

    print(f"Total images: {len(original_images)}")
    print(f"Total image list: {len(images_list)}")
    print(f"Train dataset size: {len(train_images)}")
    test_images = [os.path.join(original_dataset_dir, 
                              img.split('-')[0], 
                              img.split('-')[1].split('.')[0] + '.jpg')  # Extract original filename
                 for img in images_list[test_index]]
    print(f"Test dataset size: {len(test_images)}")


    train_ds = tf.data.Dataset.from_tensor_slices(train_images)

    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    train_ds = configure_for_performance(train_ds)

    test_ds = tf.data.Dataset.from_tensor_slices(test_images)

    test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    test_ds = configure_for_performance(test_ds)

    train(train_ds,
          test_ds=test_ds,  # Pass as test dataset
          name=f'fold_{i}')
