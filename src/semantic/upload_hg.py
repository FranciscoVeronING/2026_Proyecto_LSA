import os
from huggingface_hub import HfApi, create_repo, create_tag

# Configuración
LOCAL_FOLDER = "src/semantic/outputs/unsloth_Qwen2.5-3B-Instruct"  # Ruta donde está tu modelo entrenado
REPO_ID = "maite123/semantic-v2"        # Tu usuario/nombre_del_repo en Hugging Face
# HF_TOKEN = os.getenv("HF_TOKEN")            # Token con permisos de 'Write'
PRIVATE = False  
VERSION_TAG = "v2.0"

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
        commit_message="Release v3.0: ampliacion del dataset. Metricas"
    )

    print(f"Creando tag {VERSION_TAG}...")
    create_tag(
        repo_id=REPO_ID,
        tag=VERSION_TAG,
        tag_message="Versión 3.0 con métricas >95% en BLEU, ROUGE-L y METEOR",
        repo_type="model",
        # token=HF_TOKEN,
        exist_ok=True,
    )
    print(f"✅ Versión {VERSION_TAG} publicada en: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    upload()