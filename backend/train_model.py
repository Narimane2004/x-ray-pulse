# Generated from: DENSENET121.ipynb
# Converted at: 2026-05-09T23:27:09.462Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


train_dir = r"C:\pneumoniee\chest_xray\chest_xray\train"
test_dir  = r"C:\pneumoniee\chest_xray\chest_xray\test"
BATCH_SIZE    = 32
EPOCHS        = 10
IMG_SIZE      = (224, 224)
LEARNING_RATE = 1e-4

for d in [train_dir, test_dir]:
    if not os.path.exists(d):
        raise FileNotFoundError(f'Dossier introuvable : {d}')
    print(f'Dossier trouvé : {d}')

import os
import pandas as pd

data = []

for label in os.listdir(train_dir):
    class_dir = os.path.join(train_dir, label)
    
    if os.path.isdir(class_dir):
        for img in os.listdir(class_dir):
            data.append([os.path.join(class_dir, img), label])

df = pd.DataFrame(data, columns=["Filename", "label"])


import matplotlib.pyplot as plt
import cv2


print(df['label'].value_counts())

df['label'].value_counts().plot(kind='bar', title="Répartition des classes")
plt.show()

sample = df.sample(6)
plt.figure(figsize=(10,6))

for i, row in enumerate(sample.itertuples()):
    img = cv2.imread(row.Filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(2,3,i+1)
    plt.imshow(img)
    plt.title(row.label)
    plt.axis('off')

plt.show()



train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

categories = list(train_generator.class_indices.keys())

print(f'Classes détectées   : {train_generator.class_indices}')
print(f'Échantillons train  : {train_generator.samples}')
print(f'Échantillons val    : {val_data.samples}')
print(f'Échantillons test   : {test_generator.samples}')

cls_train = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(cls_train),
    y=cls_train
)
class_weights_dict = dict(enumerate(class_weights))
print(f'Class Weights : {class_weights_dict}')

    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
    )

model.summary()

save_path = os.path.join(os.getcwd(), 'densenet_model.keras')

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-6),
    ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
]

print(f'Modèle sauvegardé dans : {save_path}')

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_data,
    class_weight=class_weights_dict,
    callbacks=callbacks,
    verbose=1
)

acc      = history.history['accuracy']
val_acc  = history.history['val_accuracy']
loss     = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc,     label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Val Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss,     label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.legend(loc='upper right')
plt.title('Loss')
plt.tight_layout()
plt.show()

test_results = model.evaluate(test_generator)
print(f'Test Accuracy : {test_results[1]*100:.2f}%')
print(f'Test Recall   : {test_results[2]*100:.2f}%')

test_generator.reset()
preds_test        = model.predict(test_generator)
predicted_classes = (preds_test > 0.5).astype('int32').flatten()
true_classes      = test_generator.classes

print('\nClassification Report (Test):\n')
print(classification_report(true_classes, predicted_classes, target_names=categories))

print('Confusion Matrix')
print(confusion_matrix(true_classes, predicted_classes))