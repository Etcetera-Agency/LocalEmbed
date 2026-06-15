from fastembed import TextEmbedding
from loguru import logger
from app.api.schemas.list_models import ModelInfo, ModelType
from app.services.custom_models import register_custom_models

try:
    register_custom_models()
    dense_models = TextEmbedding.list_supported_models()
    logger.info(f"Successfully loaded {len(dense_models)} supported models")
except Exception as e:
    logger.error(f"Error fetching supported models: {e}")
    dense_models = []


dense_model_data = [
    ModelInfo(
        id=m["model"],
        type=ModelType.DENSE,
        dimensions=m.get("dim"),
        description=m.get("description"),
        license=m.get("license"),
        size_in_GB=m.get("size_in_GB"),
    )
    for m in dense_models
]


def get_dense_models() -> list[ModelInfo]:
    """Return a list of supported dense embedding models with their metadata."""
    return dense_model_data


def validate_model_id(model_id: str) -> bool:
    """validate that the provided model_id exists in our supported models"""
    return any(model.id == model_id for model in dense_model_data)
