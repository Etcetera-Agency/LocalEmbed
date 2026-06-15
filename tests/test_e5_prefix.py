import pytest

from app.services import embedder


def test_e5_defaults_to_passage_prefix(monkeypatch):
    calls = []

    class Model:
        def embed(self, texts, batch_size):
            calls.append(list(texts))
            return [Vector()]

        def token_count(self, texts):
            return len(list(texts))

    class Vector:
        def tolist(self):
            return [1.0, 0.0, 0.0]

    monkeypatch.setattr(embedder, "get_model", lambda _model_id: Model())

    result = embedder.embed_text(["Привіт"], model_id="intfloat/multilingual-e5-large")

    assert calls == [["passage: Привіт"]]
    assert result.vectors == [[1.0, 0.0, 0.0]]


def test_e5_query_alias_adds_query_prefix(monkeypatch):
    calls = []

    class Model:
        def embed(self, texts, batch_size):
            calls.append(list(texts))
            return [Vector()]

        def token_count(self, texts):
            return len(list(texts))

    class Vector:
        def tolist(self):
            return [1.0, 0.0, 0.0]

    monkeypatch.setattr(embedder, "get_model", lambda _model_id: Model())

    model_id, input_type = embedder.resolve_model_request(
        "intfloat/multilingual-e5-large:query"
    )
    result = embedder.embed_text(["де договір?"], model_id=model_id, input_type=input_type)

    assert model_id == "intfloat/multilingual-e5-large"
    assert calls == [["query: де договір?"]]
    assert result.model_used == "intfloat/multilingual-e5-large"


def test_existing_e5_prefix_is_preserved(monkeypatch):
    calls = []

    class Model:
        def embed(self, texts, batch_size):
            calls.append(list(texts))
            return [Vector()]

        def token_count(self, texts):
            return len(list(texts))

    class Vector:
        def tolist(self):
            return [1.0, 0.0, 0.0]

    monkeypatch.setattr(embedder, "get_model", lambda _model_id: Model())

    embedder.embed_text(
        ["query: де договір?"],
        model_id="intfloat/multilingual-e5-large",
        input_type="passage",
    )

    assert calls == [["query: де договір?"]]


def test_invalid_input_type_rejected():
    with pytest.raises(ValueError):
        embedder.resolve_model_request("intfloat/multilingual-e5-large", "bad")
