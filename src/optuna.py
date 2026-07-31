import os
import json
import gc
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import train_test_split
import optuna
import matplotlib.pyplot as plt

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
    FRAME_FEATURES_DIM
)


def load_dataset():
    """Carga y valida los arrays guardados."""
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
    return X_data, y_data, class_a_index


def create_model_from_params(params, in_channels):
    """Construye y compila el modelo recibiendo un diccionario de hiperparámetros e in_channels."""
    if 'in_channels' in get_model.__code__.co_varnames:
        base_model = get_model(in_channels=in_channels)
    else:
        base_model = get_model()

    base_model.load_weights(WEIGHTS_PATH, by_name=True, skip_mismatch=True)
    base_model.trainable = True

    # Tomamos la salida antes de la última capa del backbone
    x = base_model.layers[-2].output

    # Capa Dense intermedia si fue sugerida
    if params.get("use_dense", False):
        x = layers.Dense(params["dense_units"], activation="relu")(x)
        x = layers.Dropout(params["dropout_rate"])(x)

    # Capa final de clasificación
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='lsa_classifier_94')(x)
    model = keras.Model(inputs=base_model.input, outputs=outputs)

    learning_rate = params["learning_rate"]
    optimizer_name = params["optimizer"]

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


def create_model(trial, in_channels):
    """Sugiere hiperparámetros desde un Trial de Optuna y construye el modelo."""
    params = {
        "use_dense": trial.suggest_categorical("use_dense", [True, False]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw", "rmsprop"])
    }

    if params["use_dense"]:
        params["dense_units"] = trial.suggest_int("dense_units", 64, 512, step=64)
        params["dropout_rate"] = trial.suggest_float("dropout_rate", 0.1, 0.5)

    return create_model_from_params(params, in_channels=in_channels)


def objective(trial, X_train, y_train, X_val, y_val):
    """Función objetivo que entrena la red y retorna la métrica a optimizar."""
    keras.backend.clear_session()
    gc.collect()

    in_channels = X_train.shape[-1]
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    model = create_model(trial, in_channels=in_channels)

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

    callbacks = [
        CustomOptunaPruningCallback(trial, monitor="val_accuracy"),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=0
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
        keras.backend.clear_session()
        gc.collect()

    return val_accuracy


def plot_and_save_history(history, save_dir):
    """Genera y guarda el gráfico de Accuracy y Loss en un archivo PNG para el informe."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico de Accuracy
    ax1.plot(epochs_range, acc, label='Entrenamiento', color='tab:blue', linewidth=2)
    ax1.plot(epochs_range, val_acc, label='Validación', color='tab:orange', linewidth=2)
    ax1.set_title('Precisión (Accuracy)')
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # Gráfico de Loss
    ax2.plot(epochs_range, loss, label='Entrenamiento', color='tab:blue', linewidth=2)
    ax2.plot(epochs_range, val_loss, label='Validación', color='tab:orange', linewidth=2)
    ax2.set_title('Función de Pérdida (Loss)')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('Loss')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    output_path = os.path.join(save_dir, "best_model_learning_curves.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\n📊 Gráfico de curvas guardado con éxito en: {output_path}")


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

    # Usamos un nombre de estudio específico para el dataset actual
    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        study_name="lsa_pose_hands_no_face",
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

    # 1. Guardar los mejores hiperparámetros en JSON
    best_params_path = os.path.join(MODEL_SAVE_DIR, "best_hyperparameters.json")
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=4)
    print(f"\nMejores hiperparámetros guardados en: {best_params_path}")

    # 2. Entrenar el modelo final con la combinación ganadora
    print("\n🚀 Entrenando el MODELO FINAL con los mejores hiperparámetros...")
    in_channels = X_train.shape[-1]
    best_model = create_model_from_params(study.best_params, in_channels=in_channels)

    final_callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
    ]

    history = best_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=study.best_params["batch_size"],
        callbacks=final_callbacks,
        verbose=1
    )

    # 3. Generar y guardar el gráfico PNG para el informe
    plot_and_save_history(history, MODEL_SAVE_DIR)

    # 4. Guardar los pesos finales
    final_weights_path = os.path.join(MODEL_SAVE_DIR, "best_optuna_model.h5")
    best_model.save_weights(final_weights_path)
    print(f"💾 Pesos del modelo mejorado guardados en: {final_weights_path}")