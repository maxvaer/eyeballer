#!/usr/bin/env python3
"""One-time migration: convert Keras 2 weights to Keras 3 format.

Run this with Python 3.11 + TensorFlow 2.15 (Keras 2):

    python3.11 -m pip install tensorflow==2.15 Augmentor pandas
    python3.11 convert_weights.py

The output file eyeballer-v3.weights.h5 can then be used with
TensorFlow >= 2.16 (Keras 3) and committed to the repo.
"""

import sys
import tensorflow as tf

print(f"TensorFlow {tf.__version__}")
if not tf.__version__.startswith("2.15"):
    print("WARNING: This script is intended for TensorFlow 2.15 (Keras 2).")
    print("         Running on a different version may produce unusable weights.")
    response = input("Continue anyway? [y/N] ")
    if response.strip().lower() != "y":
        sys.exit(1)

DATA_LABELS = ["custom404", "login", "webapp", "oldlooking", "parked"]
IMAGE_WIDTH, IMAGE_HEIGHT = 224, 224
INPUT_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)

# Build the same architecture used in EyeballModel
model = tf.keras.Sequential()
pretrained_layer = tf.keras.applications.mobilenet.MobileNet(
    weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
model.add(pretrained_layer)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(256, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(len(DATA_LABELS), activation="sigmoid"))

model.compile(optimizer=tf.keras.optimizers.Adam(0.0005),
              loss="binary_crossentropy",
              metrics=["accuracy"])

src = "bishop-fox-pretrained-v3.h5"
dst = "eyeballer-v3.weights.h5"

print(f"Loading weights from {src} ...")
model.load_weights(src)
print("Loaded successfully.")

print(f"Saving in Keras 3 format to {dst} ...")
model.save_weights(dst)
print("Done. Commit eyeballer-v3.weights.h5 to your fork.")
