from fastembed import TextEmbedding

from app.services.custom_models import register_custom_models


def test_multilingual_e5_small_is_registered():
    register_custom_models()

    models = {
        model["model"]: model
        for model in TextEmbedding.list_supported_models()
    }

    model = models["intfloat/multilingual-e5-small"]
    assert model["dim"] == 384
    assert model["model_file"] == "onnx/model_int8.onnx"

