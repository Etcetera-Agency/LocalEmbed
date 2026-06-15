from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType
from loguru import logger

CUSTOM_MODELS = (
    {
        "model": "intfloat/multilingual-e5-small",
        "pooling": PoolingType.MEAN,
        "normalization": True,
        "sources": ModelSource(hf="Teradata/multilingual-e5-small"),
        "dim": 384,
        "model_file": "onnx/model_int8.onnx",
        "description": (
            "Text embeddings, Unimodal (text), Multilingual (~100 languages), "
            "512 input tokens truncation, Prefixes for queries/documents: necessary."
        ),
        "license": "mit",
        "size_in_gb": 0.113,
        "additional_files": [
            "config.json",
            "sentencepiece.bpe.model",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
    },
)


def register_custom_models() -> None:
    registered = {
        model["model"].lower() for model in TextEmbedding.list_supported_models()
    }
    for model in CUSTOM_MODELS:
        model_name = model["model"]
        if model_name.lower() in registered:
            continue

        TextEmbedding.add_custom_model(**model)
        logger.info(f"Registered custom embedding model: {model_name}")
