import os
import json
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from src.model.backbone import get_model
from src.config import SIGN_CLASSES, LEARNING_RATE_TRANSFER, DATASET_NPY_DIR, WEIGHTS_PATH, \
    NUM_CLASSES, MODEL_SAVE_DIR, BATCH_SIZE, EPOCHS, PATIENCE, VAL_SIZE

X_data = []
y_data = []

class_a_index = {class_name: i for i, class_name in enumerate(SIGN_CLASSES)}

for class_name in SIGN_CLASSES:
    class_path = os.path.join(DATASET_NPY_DIR, class_name)
    if os.path.isdir(class_path):
        for npy_file in os.listdir(class_path):
            file_path = os.path.join(class_path, npy_file)
            
            landmarks = np.load(file_path) 
            X_data.append(landmarks)
            y_data.append(class_a_index[class_name])

X_data = np.array(X_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.int32)

print(f"Dataset loaded sucessfully. Total samples: {len(X_data)}")

# 80% Trainning y 20% Validation
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=VAL_SIZE, random_state=42, stratify=y_data)

# TRANSFER LEARNING
print("Initializing base model...")
# create the base model with 250 classes, which will be replaced by a custom classifier for 94 classes
base_model = get_model() 

print("Adding weights from pre-trained model...")
base_model.load_weights(WEIGHTS_PATH, by_name=True, skip_mismatch=True)

base_model.trainable = True

# Remove the last layer of the base model to prepare for our custom classifier
x = base_model.layers[-2].output

# Add a new Dense layer for our specific classification task (94 classes)
outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='lsa_classifier_94')(x)

# Create the new model that combines the base model and our custom classifier
model_lsa = keras.Model(inputs=base_model.input, outputs=outputs)

# COMPILE THE MODEL
model_lsa.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=LEARNING_RATE_TRANSFER),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(model_lsa.summary())

# Callbacks to save the best model and implement early stopping
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
model_path = os.path.join(MODEL_SAVE_DIR, 'lsa_transfer_best.h5')
callbacks = [
    keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', mode='max'),
    keras.callbacks.EarlyStopping(patience=PATIENCE, monitor='val_loss', restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,       # Multiplies the LR by 0.2 (reduces it by 80%)
        patience=5,        # Waits 5 epochs of "plateau" before acting.             TRY 5 = 3
        min_lr=1e-6,       # The minimum LR that can be reached
        verbose=1          # Informs you in the terminal when the change is made
    )
]

print("Initiating training...")
history = model_lsa.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

model_lsa.save(model_path)
print("Best model saved at: " + model_path)

mapping_path = os.path.join(MODEL_SAVE_DIR, "mapeo_clases.json")
with open(mapping_path, "w", encoding="utf-8") as f:
    json.dump(class_a_index, f, ensure_ascii=False, indent=2)
print("Class mapping saved at: " + mapping_path)


print("Generating reports...")

REPORT_DIR = os.path.join(MODEL_SAVE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# A. Loss y Accuracy
plt.figure(figsize=(14, 5))

# Subplot 1: Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color='blue', lw=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='red', linestyle='--', lw=2)
plt.title("Curva de Pérdida (Loss) - LSA Transformer")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.legend()
plt.grid(True)

# Subplot 2: Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue', lw=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red', linestyle='--', lw=2)
plt.title("Accuracy - LSA Transfer Learning Media Pipe")
plt.xlabel("Epochs")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "accuracy_lsa.png"), dpi=300)
plt.close()


print("Calculating predictions...")
y_pred_prob = model_lsa.predict(X_val, batch_size=BATCH_SIZE)
y_pred = np.argmax(y_pred_prob, axis=1)

cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(24, 20))
sns.heatmap(
    cm, 
    annot=False,
    cmap='Blues', 
    xticklabels=SIGN_CLASSES, 
    yticklabels=SIGN_CLASSES
)
plt.title("Confusion Matrix - 94 Classes LSA", fontsize=16)
plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("True Class", fontsize=12)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()

plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix_lsa.png"), dpi=300)
plt.close()

classes_index = list(range(NUM_CLASSES))

report_txt = classification_report(
    y_val, 
    y_pred, 
    labels=classes_index, 
    target_names=SIGN_CLASSES
)
report_path = os.path.join(REPORT_DIR, "report.txt")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("=== REPORT - TRANSFER LEARNING MEDIA PIPE LSA ===\n\n")
    f.write(report_txt)
