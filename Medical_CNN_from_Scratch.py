# -*- coding: utf-8 -*-
"""
Brain Tumor Diagnostic AI - Custom CNN (Scratch)
This script builds, tunes, and trains a Convolutional Neural Network from scratch
to classify 4 types of brain tumors from MRI scans.
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
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# PHASE 1: INFRASTRUCTURE & DATA PREP
# ==========================================
# --- 1. DOWNLOAD DATASET ---
if not os.path.exists('./Brain-Tumor-Classification-DataSet'):
    print("⬇️ Downloading Brain Tumor Dataset...")
    # Using os.system instead of Colab's ! to make this script environment-agnostic
    os.system("git clone https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet.git")
    print("Download Complete.")
else:
    print("Data already downloaded.")

# --- 2. REORGANIZE & SPLIT (Train/Val/Test) ---
source_root = './Brain-Tumor-Classification-DataSet'
dataset_dir = './brain_tumor_scratch_data'
classes = ['glioma_tumor', 'meningioma_tumor', 'pituitary_tumor', 'no_tumor']

for split in ['train', 'val', 'test']:
    for cls in classes:
        os.makedirs(os.path.join(dataset_dir, split, cls), exist_ok=True)

print("\n🔄 Organizing and Splitting Data (70% Train, 20% Val, 10% Test)...")

for cls in classes:
    all_images = []

    path_a = os.path.join(source_root, 'Training', cls)
    if os.path.exists(path_a):
        all_images.extend([os.path.join(path_a, f) for f in os.listdir(path_a)])

    path_b = os.path.join(source_root, 'Testing', cls)
    if os.path.exists(path_b):
        all_images.extend([os.path.join(path_b, f) for f in os.listdir(path_b)])

    random.shuffle(all_images)
    total = len(all_images)
    train_end = int(total * 0.7)
    val_end = int(total * 0.9)

    train_imgs = all_images[:train_end]
    val_imgs = all_images[train_end:val_end]
    test_imgs = all_images[val_end:]

    for src_path in train_imgs:
        shutil.copy(src_path, os.path.join(dataset_dir, 'train', cls, os.path.basename(src_path)))
    for src_path in val_imgs:
        shutil.copy(src_path, os.path.join(dataset_dir, 'val', cls, os.path.basename(src_path)))
    for src_path in test_imgs:
        shutil.copy(src_path, os.path.join(dataset_dir, 'test', cls, os.path.basename(src_path)))

    print(f"    -> {cls}: {len(train_imgs)} Train | {len(val_imgs)} Val | {len(test_imgs)} Test")

# --- 3. DATA GENERATORS (SCRATCH CONFIG) ---
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

print("\n⚙️ Configuring Generators (Rescale 1/255)...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb'
)

val_generator = val_test_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb'
)

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False
)

print("✅ Phase 1 Complete: Medical Data Ready for Scratch Model.")

# ==========================================
# PHASE 2: EXPERIMENT ENGINE (HYPERPARAMETER TUNING)
# ==========================================
DEFAULT_FILTERS = 32
DEFAULT_DROPOUT = 0.5
DEFAULT_BLOCKS = 3
DEFAULT_KERNEL = (3, 3)
DEFAULT_BATCH = 32
DEFAULT_LR = 0.001

def run_scratch_experiment(start_filters, dropout_rate, num_blocks, kernel_size):
    model = models.Sequential()
    current_filters = start_filters

    # Block 1
    model.add(layers.Conv2D(current_filters, kernel_size, activation='relu', padding='same', input_shape=(224, 224, 3)))\
    model.add(layers.MaxPooling2D((2, 2)))

    # Additional Blocks
    for i in range(1, num_blocks):
        current_filters *= 2
        model.add(layers.Conv2D(current_filters, kernel_size, activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))

    # Head
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(4, activation='softmax'))

    model.compile(optimizer=optimizers.Adam(learning_rate=DEFAULT_LR),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"\n🔄 TEST: Filters={start_filters} | Drop={dropout_rate} | Blocks={num_blocks} | Ker={kernel_size}")
    early_stop = EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True, verbose=0)

    start_time = time.time()
    hist = model.fit(train_generator, epochs=10, validation_data=val_generator, callbacks=[early_stop], verbose=1)

    best_acc = max(hist.history['val_accuracy'])
    return best_acc, round(time.time() - start_time, 2)

# STAGE 1: FILTER QUANTITY (Width)
print(f"\n🚀 STAGE 1: Testing Model Width...")
results_s1 = []
for f in [32, 64]:
    acc, duration = run_scratch_experiment(start_filters=f, dropout_rate=0.5, num_blocks=3, kernel_size=(3,3))
    results_s1.append({'Filters': f, 'Val Acc': acc, 'Time': duration})

df1 = pd.DataFrame(results_s1)
BEST_FILTERS = int(df1.loc[df1['Val Acc'].idxmax()]['Filters'])
print(f"🏆 WINNER: {BEST_FILTERS} Filters")

# STAGE 2: DROPOUT (Regularization)
print(f"\n🚀 STAGE 2: Testing Regularization...")
results_s2 = []
for d in [0.0, 0.5]:
    acc, duration = run_scratch_experiment(start_filters=BEST_FILTERS, dropout_rate=d, num_blocks=3, kernel_size=(3,3))
    results_s2.append({'Dropout': d, 'Val Acc': acc, 'Time': duration})

df2 = pd.DataFrame(results_s2)
BEST_DROPOUT = float(df2.loc[df2['Val Acc'].idxmax()]['Dropout'])
print(f"🏆 WINNER: Dropout {BEST_DROPOUT}")

# STAGE 3: NETWORK DEPTH
print(f"\n🚀 STAGE 3: Testing Network Depth...")
results_s3 = []
for blocks in [2, 3]:
    acc, duration = run_scratch_experiment(start_filters=BEST_FILTERS, dropout_rate=BEST_DROPOUT, num_blocks=blocks, kernel_size=(3,3))
    results_s3.append({'Blocks': blocks, 'Val Acc': acc, 'Time': duration})

df3 = pd.DataFrame(results_s3)
BEST_BLOCKS = int(df3.loc[df3['Val Acc'].idxmax()]['Blocks'])
print(f"🏆 WINNER: {BEST_BLOCKS} Blocks")

# ==========================================
# PHASE 3: FINAL REPORT & GRAND CHAMPION
# ==========================================
print("\n🏆 OPTIMIZATION TOURNAMENT RESULTS")
print("="*40)
print(f"1. Best Width:    {BEST_FILTERS} Filters")
print(f"2. Best Dropout:  {BEST_DROPOUT}")
print(f"3. Best Depth:    {BEST_BLOCKS} Blocks")
print("="*40)

model_champion = models.Sequential(name="Grand_Champion_Scratch")
curr_filters = BEST_FILTERS

model_champion.add(layers.Conv2D(curr_filters, (3,3), activation='relu', padding='same', input_shape=(224, 224, 3)))
model_champion.add(layers.MaxPooling2D((2, 2)))

for i in range(1, BEST_BLOCKS):
    curr_filters *= 2
    model_champion.add(layers.Conv2D(curr_filters, (3,3), activation='relu', padding='same'))
    model_champion.add(layers.MaxPooling2D((2, 2)))

model_champion.add(layers.Flatten())
model_champion.add(layers.Dense(256, activation='relu'))
model_champion.add(layers.Dropout(BEST_DROPOUT))
model_champion.add(layers.Dense(4, activation='softmax'))

model_champion.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                       loss='categorical_crossentropy', metrics=['accuracy'])

print(f"\n🚀 TRAINING GRAND CHAMPION MODEL...")
early_stop = EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True)
history_final = model_champion.fit(train_generator, epochs=25, validation_data=val_generator, callbacks=[early_stop], verbose=1)

print("\n📝 FINAL TEST EVALUATION")
test_loss, test_acc = model_champion.evaluate(test_generator)
print(f"🏆 FINAL ACCURACY: {test_acc*100:.2f}%")

print("\n📊 GENERATING MEDICAL DIAGNOSTIC REPORT...")
test_generator.reset()
predictions = model_champion.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print("\n📑 DETAILED CLASSIFICATION METRICS:\n")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

model_champion.save('final_medical_scratch_model.h5')
