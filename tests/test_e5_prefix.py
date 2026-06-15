import pytest

from app.services import embedder


class Vector:
    def tolist(self):
        return [1.0, 0.0, 0.0]


class RecordingModel:
    def __init__(self, calls):
        self.calls = calls

    def embed(self, texts, batch_size):
        self.calls.append(list(texts))
        return [Vector()]

    def token_count(self, texts):
        return len(list(texts))


def stub_model(monkeypatch):
    calls = []
    monkeypatch.setattr(embedder, "get_model", lambda _model_id: RecordingModel(calls))
    return calls


def test_e5_defaults_to_passage_prefix(monkeypatch):
    calls = stub_model(monkeypatch)

    result = embedder.embed_text(["Привіт"], model_id="intfloat/multilingual-e5-large")

    assert calls == [["passage: Привіт"]]
    assert result.vectors == [[1.0, 0.0, 0.0]]


def test_e5_auto_prefix_treats_short_question_as_query(monkeypatch):
    calls = stub_model(monkeypatch)
    monkeypatch.setattr(embedder.settings, "DEFAULT_INPUT_TYPE", "auto")

    embedder.embed_text(["де договір?"], model_id="intfloat/multilingual-e5-large")

    assert calls == [["query: де договір?"]]


def test_e5_auto_prefix_treats_long_text_as_passage(monkeypatch):
    calls = stub_model(monkeypatch)
    text = (
        "Цей фрагмент документа описує порядок погодження договору, строки "
        "відповідальних осіб та правила зберігання фінальної версії після підписання."
    )
    monkeypatch.setattr(embedder.settings, "DEFAULT_INPUT_TYPE", "auto")

    embedder.embed_text([text], model_id="intfloat/multilingual-e5-large")

    assert calls == [[f"passage: {text}"]]


def test_e5_auto_prefix_treats_short_memory_statement_as_passage(monkeypatch):
    calls = stub_model(monkeypatch)
    text = "User prefers Romanian examples with Cyrillic explanations"
    monkeypatch.setattr(embedder.settings, "DEFAULT_INPUT_TYPE", "auto")

    embedder.embed_text([text], model_id="intfloat/multilingual-e5-large")

    assert calls == [[f"passage: {text}"]]


def test_e5_explicit_input_type_overrides_auto_default(monkeypatch):
    calls = stub_model(monkeypatch)
    monkeypatch.setattr(embedder.settings, "DEFAULT_INPUT_TYPE", "auto")

    embedder.embed_text(
        ["де договір?"],
        model_id="intfloat/multilingual-e5-large",
        input_type="passage",
    )

    assert calls == [["passage: де договір?"]]


def test_e5_query_alias_adds_query_prefix(monkeypatch):
    calls = stub_model(monkeypatch)

    model_id, input_type = embedder.resolve_model_request(
        "intfloat/multilingual-e5-large:query"
    )
    result = embedder.embed_text(["де договір?"], model_id=model_id, input_type=input_type)

    assert model_id == "intfloat/multilingual-e5-large"
    assert calls == [["query: де договір?"]]
    assert result.model_used == "intfloat/multilingual-e5-large"


def test_existing_e5_prefix_is_preserved(monkeypatch):
    calls = stub_model(monkeypatch)

    embedder.embed_text(
        ["query: де договір?"],
        model_id="intfloat/multilingual-e5-large",
        input_type="passage",
    )

    assert calls == [["query: де договір?"]]


def test_invalid_input_type_rejected():
    with pytest.raises(ValueError):
        embedder.resolve_model_request("intfloat/multilingual-e5-large", "bad")
