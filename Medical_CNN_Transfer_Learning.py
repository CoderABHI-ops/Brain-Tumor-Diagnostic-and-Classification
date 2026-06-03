# -*- coding: utf-8 -*-
"""
Brain Tumor Diagnostic AI - Transfer Learning (VGG16)
This script utilizes a pre-trained VGG16 architecture, implementing fine-tuning
and dynamic hyperparameter search to classify medical MRI scans.
"""

import os
import shutil
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# PHASE 1: INFRASTRUCTURE & DATA PREP
# ==========================================
if not os.path.exists('./Brain-Tumor-Classification-DataSet'):
    print("⬇️ Downloading Brain Tumor Dataset...")
    os.system("git clone https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet.git")
    print("Download Complete.")
else:
    print("Data already downloaded.")

source_root = './Brain-Tumor-Classification-DataSet'
dataset_dir = './brain_tumor_transfer_data'
classes = ['glioma_tumor', 'meningioma_tumor', 'pituitary_tumor', 'no_tumor']

for split in ['train', 'val', 'test']:\
    for cls in classes:
        os.makedirs(os.path.join(dataset_dir, split, cls), exist_ok=True)

print("\n🔄 Organizing and Splitting Data...")

for cls in classes:
    all_images = []
    for folder in ['Training', 'Testing']:
        src_path = os.path.join(source_root, folder, cls)
        if os.path.exists(src_path):
            all_images.extend([os.path.join(src_path, f) for f in os.listdir(src_path)])

    random.shuffle(all_images)
    total = len(all_images)
    train_end = int(total * 0.7)
    val_end = int(total * 0.9)

    for i, path in enumerate(all_images):
        fname = os.path.basename(path)
        if i < train_end:
            shutil.copy(path, os.path.join(dataset_dir, 'train', cls, fname))
        elif i < val_end:
            shutil.copy(path, os.path.join(dataset_dir, 'val', cls, fname))
        else:
            shutil.copy(path, os.path.join(dataset_dir, 'test', cls, fname))

BATCH_SIZE = 32
IMG_SIZE = (224, 224)

print("\n⚙️ Configuring Generators (With VGG Preprocessing)...")
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15, zoom_range=0.1, horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'train'), target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'val'), target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'test'), target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

# ==========================================
# PHASE 2: EXPERIMENT ENGINE
# ==========================================
IMG_SHAPE = (224, 224, 3)
NUM_CLASSES = 4

def run_transfer_experiment(dense_units, dropout_rate, fine_tune_at, learning_rate):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)
    base_model.trainable = False

    if fine_tune_at > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune_at]:
            layer.trainable = False

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"\n🔄 TEST: Units={dense_units} | Drop={dropout_rate} | Unfreeze={fine_tune_at} | LR={learning_rate}")
    early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=0)

    start_t = time.time()
    hist = model.fit(train_generator, epochs=10, validation_data=val_generator, callbacks=[early_stop], verbose=1)
    return max(hist.history['val_accuracy']), round(time.time() - start_t, 2)

# STAGE 1: CLASSIFIER WIDTH
print(f"\n🚀 STAGE 1: Testing Classifier Width...")
results_t1 = []
for units in [256, 512]:
    acc, duration = run_transfer_experiment(units, 0.5, 0, 0.001)
    results_t1.append({'Units': units, 'Val Acc': acc})

df1 = pd.DataFrame(results_t1)
BEST_UNITS = int(df1.loc[df1['Val Acc'].idxmax()]['Units'])
print(f"🏆 WINNER: {BEST_UNITS} Neurons")

# STAGE 2: DROPOUT
print(f"\n🚀 STAGE 2: Testing Dropout...")
results_t2 = []
for drop in [0.3, 0.5]:
    acc, duration = run_transfer_experiment(BEST_UNITS, drop, 0, 0.001)
    results_t2.append({'Dropout': drop, 'Val Acc': acc})

df2 = pd.DataFrame(results_t2)
BEST_DROPOUT = float(df2.loc[df2['Val Acc'].idxmax()]['Dropout'])
print(f"🏆 WINNER: Dropout {BEST_DROPOUT}")

# STAGE 3: FINE-TUNING
print(f"\n🚀 STAGE 3: Testing Fine-Tuning (Frozen vs Unfrozen)...")
results_t3 = []
acc_frozen, _ = run_transfer_experiment(BEST_UNITS, BEST_DROPOUT, 0, 0.001)
results_t3.append({'Type': 'Frozen', 'Unfreeze_Layers': 0, 'LR': 0.001, 'Val Acc': acc_frozen})

acc_tuned, _ = run_transfer_experiment(BEST_UNITS, BEST_DROPOUT, 4, 0.0001)
results_t3.append({'Type': 'Fine-Tuned', 'Unfreeze_Layers': 4, 'LR': 0.0001, 'Val Acc': acc_tuned})

df3 = pd.DataFrame(results_t3)
best_row = df3.loc[df3['Val Acc'].idxmax()]
BEST_UNFREEZE = int(best_row['Unfreeze_Layers'])
BEST_LR = float(best_row['LR'])
print(f"🏆 STAGE 3 WINNER: {best_row['Type']} (Acc: {best_row['Val Acc']:.4f})")

# ==========================================
# PHASE 3: GRAND CHAMPION (TRANSFER LEARNING)
# ==========================================
print(f"\n🏆 CONSTRUCTING GRAND CHAMPION MODEL")
print(f"   -> Mode: {'Fine-Tuned' if BEST_UNFREEZE > 0 else 'Frozen'}")
print(f"   -> Hyperparams: {BEST_UNITS} Units | {BEST_DROPOUT} Dropout | LR {BEST_LR}")

base_model = VGG16(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)
base_model.trainable = False
if BEST_UNFREEZE > 0:
    base_model.trainable = True
    for layer in base_model.layers[:-BEST_UNFREEZE]:
        layer.trainable = False

model_champion = models.Sequential(name="VGG16_Grand_Champion")
model_champion.add(base_model)
model_champion.add(layers.Flatten())
model_champion.add(layers.Dense(BEST_UNITS, activation='relu'))
model_champion.add(layers.Dropout(BEST_DROPOUT))
model_champion.add(layers.Dense(NUM_CLASSES, activation='softmax'))

model_champion.compile(optimizer=optimizers.Adam(learning_rate=BEST_LR),
                       loss='categorical_crossentropy', metrics=['accuracy'])

print(f"\n🚀 TRAINING CHAMPION...")
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
history_final = model_champion.fit(train_generator, epochs=20, validation_data=val_generator, callbacks=[early_stop], verbose=1)

print("\n📊 GENERATING DIAGNOSTIC REPORT...")
test_generator.reset()
predictions = model_champion.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print("\n📑 DETAILED METRICS:\n")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

model_champion.save('final_brain_tumor_vgg16.h5')
print("✅ Model Saved to Disk.")
