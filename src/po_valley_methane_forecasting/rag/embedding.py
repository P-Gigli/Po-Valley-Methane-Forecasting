from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/"
    "multi-qa-MiniLM-L6-cos-v1"
)


def load_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> SentenceTransformer:
    """
    Load the embedding model used for semantic search.
    """
    return SentenceTransformer(model_name)


def embed_chunks(
    chunks: list[dict],
    model: SentenceTransformer,
):
    """
    Compute normalized embeddings for corpus chunks.
    """

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    return model.encode_document(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )