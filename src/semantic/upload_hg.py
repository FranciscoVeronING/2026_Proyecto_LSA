import os
from huggingface_hub import HfApi, create_repo

# Configuración
LOCAL_FOLDER = "src/semantic/outputs/unsloth_Qwen2.5-3B-Instruct"  # Ruta donde está tu modelo entrenado
REPO_ID = "maite123/semantic-v2"        # Tu usuario/nombre_del_repo en Hugging Face
# HF_TOKEN = os.getenv("HF_TOKEN")            # Token con permisos de 'Write'
PRIVATE = False                                      # Cambiar a True si querés que sea privado

def upload():
    # api = HfApi(token=HF_TOKEN)
    api = HfApi()

    # 1. Crear el repositorio si no existe
    print(f"Verificando / creando repositorio: {REPO_ID}...")
    create_repo(
        repo_id=REPO_ID,
        # token=HF_TOKEN,
        repo_type="model",
        private=PRIVATE,
        exist_ok=True,
    )

    # 2. Subir todos los archivos de la carpeta
    print(f"Subiendo archivos desde '{LOCAL_FOLDER}'...")
    api.upload_folder(
        folder_path=LOCAL_FOLDER,
        repo_id=REPO_ID,
        repo_type="model",
        # token=HF_TOKEN,
    )
    print(f"✅ Subida completada exitosamente a: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    upload()