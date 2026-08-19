import numpy as np

from collections import Counter

from po_valley_methane_forecasting.rag.reranking import(
    rerank_chunks,
)


def retrieve_candidates(
    query: str,
    model,
    corpus_embeddings,
    chunks: list[dict],
    candidate_k: int = 20,
    min_score: float = 0.35,
) -> list[dict]:

    query_embedding = model.encode_query(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    scores = (
        corpus_embeddings
        @ query_embedding
    )

    candidate_indices = np.argsort(
        scores
    )[::-1]

    results = []

    for index in candidate_indices:
        score = float(scores[index])

        if score < min_score:
            break

        results.append({
            **chunks[index],
            "score": score,
        })

        if len(results) == candidate_k:
            break

    return results


def select_final_chunks(
    chunks: list[dict],
    top_k: int = 5,
    max_per_source: int = 2,
) -> list[dict]:

    source_counts = Counter()
    results = []

    for chunk in chunks:
        source = chunk["source"]

        if (
            source_counts[source]
            >= max_per_source
        ):
            continue

        too_close = any(
            selected["source"]
            == chunk["source"]
            and selected.get("section")
            == chunk.get("section")
            and abs(
                selected["chunk_index"]
                - chunk["chunk_index"]
            ) <= 1
            for selected in results
        )

        if too_close:
            continue

        results.append(chunk)
        source_counts[source] += 1

        if len(results) == top_k:
            break

    return results


def retrieve_and_rerank(
    query,
    embedding_model,
    reranker,
    corpus_embeddings,
    chunks,
    candidate_k=20,
    top_k=5,
    min_score=0.35,
    max_per_source=2,
):

    candidates = retrieve_candidates(
        query=query,
        model=embedding_model,
        corpus_embeddings=corpus_embeddings,
        chunks=chunks,
        candidate_k=candidate_k,
        min_score=min_score,
    )

    reranked = rerank_chunks(
        query=query,
        chunks=candidates,
        model=reranker,
    )

    final_chunks = select_final_chunks(
        chunks=reranked,
        top_k=top_k,
        max_per_source=max_per_source,
    )

    return final_chunks