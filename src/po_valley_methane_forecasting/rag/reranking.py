from sentence_transformers import CrossEncoder


DEFAULT_RERANKER = (
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)


def load_reranker():
    return CrossEncoder(DEFAULT_RERANKER)


def rerank_chunks(
    query: str,
    chunks: list[dict],
    model,
) -> list[dict]:

    pairs = [
        [
            query,
            chunk.get(
                "embedding_text",
                chunk["text"],
            ),
        ]
        for chunk in chunks
    ]

    scores = model.predict(pairs)

    reranked = [
        {
            **chunk,
            "rerank_score": float(score),
        }
        for chunk, score in zip(
            chunks,
            scores,
        )
    ]

    return sorted(
        reranked,
        key=lambda x: x["rerank_score"],
        reverse=True,
    )