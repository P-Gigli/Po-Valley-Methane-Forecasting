from collections import Counter
import numpy as np


def retrieve_chunks(
    query: str,
    model,
    corpus_embeddings,
    chunks: list[dict],
    top_k: int = 5,
    max_per_source: int = 2,
    min_score: float = 0.5,
) -> list[dict]:
    """
    retrieve top chunks from documents corp via cosine similarity
    with limitation on chunks from the same source
    """
    query_embedding = model.encode_query(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    scores = corpus_embeddings @ query_embedding

    candidate_indices = np.argsort(scores)[::-1]

    source_counts = Counter()
    results = []

    for index in candidate_indices:
        score = float(scores[index])

        if score < min_score:
            break

        chunk = chunks[index]
        source = chunk["source"]

        if source_counts[source] >= max_per_source:
            continue

        too_close = any(
            selected["source"] == chunk["source"]
            and abs(
                selected["chunk_index"]
                - chunk["chunk_index"]
            ) <= 1
            for selected in results
        )

        if too_close:
            continue

        results.append({
            **chunk,
            "score": score,
        })

        source_counts[source] += 1

        if len(results) == top_k:
            break

    return results