# Cliente ILSA → FastAPI semántico en Cloud Run.
CLOUD_SEMANTIC_URL = (
    "https://ilsa-semantic-service-705664945052.us-central1.run.app"
)
DEFAULT_MODEL_ID = "remote"
SEMANTIC_TOKEN = ""


def list_semantic_models() -> list[dict]:
    return [
        {
            "id": DEFAULT_MODEL_ID,
            "label": "Cloud Run",
            "available": True,
        }
    ]
