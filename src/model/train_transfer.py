import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
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
base_model.load_weights(WEIGHTS_PATH)

# Freeze all layers of the base model to prevent them from being updated during training
for layer in base_model.layers:
    layer.trainable = False

# Remove the last layer of the base model to prepare for our custom classifier
x = base_model.layers[-2].output

# Add a new Dense layer for our specific classification task (94 classes)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name='lsa_classifier_94')(x)

# Create the new model that combines the base model and our custom classifier
model_lsa = tf.keras.Model(inputs=base_model.input, outputs=outputs)

# COMPILE THE MODEL
model_lsa.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_TRANSFER),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(model_lsa.summary())

# Callbacks to save the best model and implement early stopping
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(os.path.join(MODEL_SAVE_DIR, 'lsa_transfer_best.h5'), save_best_only=True, monitor='val_accuracy'),
    tf.keras.callbacks.EarlyStopping(patience=PATIENCE, monitor='val_loss', restore_best_weights=True)
]

print("Initiating training...")
history = model_lsa.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

print("Final model saved at: " + os.path.join(MODEL_SAVE_DIR, 'lsa_transfer_last.h5'))