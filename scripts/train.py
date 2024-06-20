from pandas._config import dates
import pandas as pd
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
import tensorflow as tf
import tensorflow_addons as tfa


msg = "Adding description"

# Initialize parser
parser = argparse.ArgumentParser(description = msg)
parser.add_argument("-o", "--output", help = "Set the output path to save the model", default="./")
parser.add_argument("-e", "--epochs", help = "Set the epochs for training", type=int, default=100)
parser.add_argument("-i", "--input", help="Set the input path for the data used for training", required=True)
parser.add_argument("-b", "--batch", help="Set the batch size", type=int, default=16)
parser.add_argument("-w", "--weights", help="Set the weights used", choices=["imagenet", "none"], default="imagenet")
parser.add_argument("-s", "--size", help="Set the input size as WIDTH HEIGHT", type=int, nargs=2, required=True)
parser.add_argument("-m", "--model", help="Set the model used", choices=['b3', 'b2', 'b1', 'b0', 'ir2'], required=True)
parser.add_argument("-k", "--k_fold", help="Set the number of folds used for Cross Validation", default=5)

args = parser.parse_args()

if args.input:
    dataset_dir = args.input
    print(f"Dataset path: {args.input}")

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
    'b0': tf.keras.applications.EfficientNetV2B0
}

if args.model:

    model = models[args.model]
    print(f"Model: {model().name}")


name = datetime.now()
name = name.strftime("%d-%m-%Y_%H:%M:%S")
name = f"{name}-{loss}-{epochs}epochs"
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
    'dataset': dataset_dir,
    'model': str(model().name),
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
    parts = tf.strings.split(file_path, os.path.sep)
    return tf.argmax(tf.strings.split(tf.strings.split(parts[-1], ".")[0], "-")[0] == codes)

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
    img = tf.cast(decode_img(img), tf.float32)
    return img, label

AUTOTUNE = tf.data.AUTOTUNE
def configure_for_performance(ds: tf.data.Dataset):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def configure_for_performance_test(ds: tf.data.Dataset):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

def test_processing(img, size):
    """
    Esta función procesa una imagen para ser evaluada por el modelo.

    @img: Una imagen en formato PIL.

    @size: Tupla de enteros que sigue el formato (width, height, n_channels)
            Ej: (300, 300, 3)

    Devuelve una imagen en formato PIL.
    """
    nimg = tf.image.random_crop(np.asarray(img), size)
    return Image.fromarray(nimg.numpy())

def test_preprocessing(img):
    return tf.keras.layers.CenterCrop(300, 300)(img)



def train(train, val=None, name=None, model=model):

    # TODO: Improve the selection of the current model
    base_model = model(
    # base_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
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

    decay_steps = int(epochs * (train.cardinality().numpy()))
    print('Decay steps:', decay_steps)
    initial_learning_rate = 0.002
    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate, decay_steps)

    adam = tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn)
    lr_metric = get_lr_metric(adam)



    model.compile(
        optimizer=adam, loss='sparse_categorical_crossentropy', metrics=["accuracy", lr_metric],
    )

    if not val:
        history = model.fit(train, epochs=epochs, batch_size=batch_size)
    else:
        history = model.fit(train, validation_data=val, epochs=epochs, batch_size=batch_size)

    if val:
        ts_score = model.evaluate(val)
        name = f'{name}-{ts_score}'

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
original_dataset = './envio-1-perfecto'
original_images = [os.path.join(original_dataset,classes, img) for classes in os.listdir(original_dataset) for img in os.listdir(os.path.join(original_dataset, classes))]
# print(images_list)
# print(labels_list)
skf = StratifiedKFold(n_splits=k_fold)
for i, (train_index, test_index) in enumerate(skf.split(images_list, labels_list)):
    train_images = [os.path.join(dataset_dir, img, i) for img in images_list[train_index] for i in os.listdir(os.path.join(dataset_dir, img))]

    print(f"Total images: {len(original_images)}")
    print(f"Total image list: {len(images_list)}")
    print(f"Train dataset size: {len(train_images)}")
    test_images = []
    for test_img in images_list[test_index]:
        for original in original_images:
            if test_img in original:
                test_images.append(original)
                break
    print(f"Test dataset size: {len(test_images)}")


    train_ds = tf.data.Dataset.from_tensor_slices(train_images)

    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    train_ds = configure_for_performance(train_ds)

    val_ds = tf.data.Dataset.from_tensor_slices(test_images)

    val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = configure_for_performance(val_ds)

    train(train_ds,
          val_ds,
          name=f'{i}')
