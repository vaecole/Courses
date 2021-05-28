import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
from distutils.dir_util import copy_tree

model_saved = './model_saved/'


def get_train_fruits():
    fruits = '''
    Pomegranate
    Kaki
    Pear
    Avocado
    Banana
    Dates
    Cocos
    Mangostan
    Lychee
    Mulberry
    
    '''.split().sort()

    # download fruit 360 dataset and put them in D:/data/fruits-360/,
    # bellow code will copy above fruits for sub-training
    # for f in fruits:
    #     copy_tree("D:/data/fruits-360/Training/"+f, "D:/data/fruits-10-high-calorie/Training/"+f)
    #     copy_tree("D:/data/fruits-360/Test/"+f, "D:/data/fruits-10-high-calorie/Test/"+f)
    return fruits


def load_data(base_path='D:/data/fruits-10-high-calorie/'):
    train_ds_ = image_dataset_from_directory(
        base_path + 'Training',
        labels='inferred',
        label_mode='categorical',
        image_size=(100, 100),
        interpolation='nearest',
        shuffle=True,
        batch_size=512
    )

    test_ds_ = image_dataset_from_directory(
        base_path + 'Test',
        labels='inferred',
        label_mode='categorical',
        image_size=(100, 100),
        interpolation='nearest',
        shuffle=False,
        batch_size=512
    )

    train_ds = (
        train_ds_
            .map(convert_to_float)
            .cache()
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    test_ds = (
        test_ds_
            .map(convert_to_float)
            .cache()
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )
    return train_ds, test_ds


def convert_to_float(image, label):
    image = image / 255
    return image, label


def get_model(final_dim):
    model = keras.Sequential([
        layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(100, 100, 3)),
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(final_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def train(is_validate=False):
    early_stopping = EarlyStopping(
        min_delta=1e-3,
        patience=5,
        restore_best_weights=True
    )
    train_ds, test_ds = load_data()
    model = get_model(train_ds._flat_shapes[1][1])
    history = model.fit(
        train_ds,
        batch_size=512,
        validation_data=test_ds,
        callbacks=[early_stopping],
        epochs=500,
        verbose=1
    )
    model.save(model_saved)
    if is_validate:
        validate(history)
    return model


def predict(img_file='D:/data/fruits-10-high-calorie/Test/Banana/111_100.jpg'):
    if os.path.isdir(model_saved):
        model = keras.models.load_model(model_saved)
    else:
        model = train()
    fruits = get_train_fruits()
    img = imread(img_file)
    predict_visual(fruits, model, img)


def validate(train_history):
    fit_hist = pd.DataFrame(train_history.history)

    loss = round(np.min(fit_hist['loss']), 2)
    val_loss = round(np.min(fit_hist['val_loss']), 2)
    acc = round(np.max(fit_hist['accuracy']), 2)
    val_acc = round(np.max(fit_hist['val_accuracy']), 2)

    plt.title(f"Train Loss ({loss}) and Validation Loss ({val_loss})")
    plt.plot(fit_hist['loss'], label='Train Loss')
    plt.plot(fit_hist['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(color='#e6e6e6')
    plt.legend()
    plt.show()

    plt.title(f"Train Accuracy ({acc}) and Validation Accuracy ({val_acc})")
    plt.plot(fit_hist['accuracy'], label='Train Acc')
    plt.plot(fit_hist['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(color='#e6e6e6')
    plt.legend()
    plt.show()


def predict_visual(fruits, model, img, show=False):
    if show:
        plt.imshow(img)
    predicted_tsr = model.predict(tf.expand_dims(img, 0))[0]
    return fruits[np.where(predicted_tsr == 1)[0][0]]


if __name__ == '__main__':
    print("start predict")
    print(predict())
    print('end')