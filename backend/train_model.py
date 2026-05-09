# Generated from: pneumonie201 (1).ipynb
# Converted at: 2026-05-09T21:18:42.338Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import DenseNet201

# =========================
# 1. AUTO DETECTION DATASET
# =========================
def find_dataset_root():
    base = "/kaggle/input"
    for root, dirs, files in os.walk(base):
        if "train" in dirs and "test" in dirs:
            print(f"Dataset trouvé : {root}")
            return root
    raise Exception("Dataset chest X-ray introuvable dans /kaggle/input")

DATA_DIR = find_dataset_root()

# =========================
# 2. CONFIG
# =========================
BATCH_SIZE_PER_REPLICA = 128
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * 2
IMG_SIZE = (224, 224)
EPOCHS = 10
LEARNING_RATE = 1e-4

print("DATA_DIR =", DATA_DIR)

# =========================
# 3. STRATEGY GPU
# =========================
strategy = tf.distribute.MirroredStrategy()
print("GPUs:", strategy.num_replicas_in_sync)

# =========================
# 4. DATA GENERATORS
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=GLOBAL_BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=GLOBAL_BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=GLOBAL_BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# =========================
# 5. CLASS WEIGHTS
# =========================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# =========================
# 6. MODEL (DenseNet201)
# =========================
with strategy.scope():

    base_model = DenseNet201(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = True
    for layer in base_model.layers[:-70]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=["accuracy", tf.keras.metrics.Recall(name="recall")]
    )

model.summary()

# =========================
# 7. CALLBACKS
# =========================
callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ModelCheckpoint("best_model.keras", monitor="val_accuracy", save_best_only=True)
]

# =========================
# 8. TRAINING
# =========================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# =========================
# 9. PLOTS
# =========================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss")
plt.legend()

plt.show()

# =========================
# 10. EVALUATION
# =========================
print("Testing...")

test_loss, test_acc, test_recall = model.evaluate(test_generator)

print("Test Accuracy:", test_acc)
print("Test Recall:", test_recall)

pred = model.predict(test_generator)
y_pred = (pred > 0.5).astype(int).flatten()
y_true = test_generator.classes

print(classification_report(y_true, y_pred))

# =========================
# 11. CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.show()