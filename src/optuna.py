import os
import json
import gc
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import train_test_split
import optuna
from optuna.integration import TFKerasPruningCallback

from src.model.backbone import get_model
from src.config import (
    SIGN_CLASSES, 
    DATASET_NPY_DIR, 
    WEIGHTS_PATH,
    NUM_CLASSES, 
    MODEL_SAVE_DIR, 
    EPOCHS, 
    PATIENCE, 
    VAL_SIZE,
    POINT_LANDMARKS
)


def load_dataset():
    """Carga y valida los arrays guardados en formato de 708 canales."""
    X_data = []
    y_data = []
    class_a_index = {class_name: i for i, class_name in enumerate(SIGN_CLASSES)}

    for class_name in SIGN_CLASSES:
        class_path = os.path.join(DATASET_NPY_DIR, class_name)
        if os.path.isdir(class_path):
            for npy_file in os.listdir(class_path):
                file_path = os.path.join(class_path, npy_file)
                landmarks = np.load(file_path)

                # Si tus .npy vinieran completos con 543 puntos, filtramos a 118 aquí:
                if landmarks.shape[-1] != 708 and landmarks.ndim == 3 and landmarks.shape[1] == 543:
                    landmarks = landmarks[:, POINT_LANDMARKS, :]
                    landmarks = landmarks.reshape(landmarks.shape[0], -1)

                X_data.append(landmarks)
                y_data.append(class_a_index[class_name])

    X_data = np.array(X_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.int32)
    return X_data, y_data, class_a_index


def create_model(trial):
    """Construye el modelo especificando 708 canales de entrada."""
    # Instanciamos el backbone especificando los 708 canales
    base_model = get_model(in_channels=708) if 'in_channels' in get_model.__code__.co_varnames else get_model()
    
    base_model.load_weights(WEIGHTS_PATH, by_name=True, skip_mismatch=True)
    base_model.trainable = True

    # Tomamos la salida antes de la última capa del backbone
    x = base_model.layers[-2].output

    # --- HIPERPARÁMETROS A SINTONIZAR ---
    
    # 1. Opción de agregar una capa Dense intermedia
    use_dense_intermedia = trial.suggest_categorical("use_dense", [True, False])
    if use_dense_intermedia:
        units = trial.suggest_int("dense_units", 64, 512, step=64)
        x = layers.Dense(units, activation="relu")(x)
        
        # 2. Rate de Dropout
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        x = layers.Dropout(dropout_rate)(x)

    # Capa final de clasificación
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='lsa_classifier_94')(x)
    
    model = keras.Model(inputs=base_model.input, outputs=outputs)

    # 3. Learning Rate (en escala logarítmica)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    
    # 4. Optimizador (Manejo de compatibilidad tf.keras)
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw", "rmsprop"])
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == "adamw":
        if hasattr(tf.keras.optimizers, "AdamW"):
            optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def objective(trial, X_train, y_train, X_val, y_val):
    """Función objetivo que entrena la red y retorna la métrica a optimizar."""
    
    # Limpieza previa de memoria GPU/RAM
    keras.backend.clear_session()
    gc.collect()

    # 5. Tamaño de Batch dinámico
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    model = create_model(trial)
    class CustomOptunaPruningCallback(keras.callbacks.Callback):
        def __init__(self, trial, monitor="val_accuracy"):
            super().__init__()
            self.trial = trial
            self.monitor = monitor

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            current_val = logs.get(self.monitor)
            if current_val is None:
                return
            
            self.trial.report(current_val, step=epoch)
            if self.trial.should_prune():
                message = f"Trial suspendido tempranamente en la epoch {epoch} por bajo rendimiento."
                raise optuna.TrialPruned(message)

    # Callbacks
    callbacks = [
        # Pruning: cancela trials con mal rendimiento anticipado
        CustomOptunaPruningCallback(trial, monitor="val_accuracy"),
        # Early Stopping
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
        # ReduceLROnPlateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=0
        )
    ]

    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        val_accuracy = max(history.history['val_accuracy'])
    finally:
        # Garantiza liberar la memoria VRAM incluso si el trial falla o es descartado
        keras.backend.clear_session()
        gc.collect()

    return val_accuracy


if __name__ == "__main__":
    print("Cargando dataset...")
    X_data, y_data, class_a_index = load_dataset()

    print(f"Shape de datos cargados: {X_data.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        X_data, y_data, test_size=VAL_SIZE, random_state=42, stratify=y_data
    )

    print(f"Datos listos: {len(X_train)} train | {len(X_val)} val")

    # Configuración de Pruning y Almacenamiento Persistente en SQLite
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    db_path = os.path.join(MODEL_SAVE_DIR, "optuna_study.db")
    storage_name = f"sqlite:///{db_path}"

    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        study_name="lsa_transfer_learning",
        storage=storage_name,
        load_if_exists=True
    )

    print("\nIniciando optimización con Optuna...")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val), 
        n_trials=30,
        catch=(Exception,)
    )

    print("\n================ MEJORES RESULTADOS ================")
    print(f"Mejor accuracy de validación: {study.best_value:.4f}")
    print("Mejores hiperparámetros encontrados:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Guardar los mejores hiperparámetros en JSON
    best_params_path = os.path.join(MODEL_SAVE_DIR, "best_hyperparameters.json")
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=4)
    print(f"\nMejores hiperparámetros guardados en: {best_params_path}")