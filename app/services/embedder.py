from fastembed import TextEmbedding

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

model = TextEmbedding(EMBEDDING_MODEL)


def embed_text(text: list[str]) -> list[list[float]]:
    return model.embed(text)
