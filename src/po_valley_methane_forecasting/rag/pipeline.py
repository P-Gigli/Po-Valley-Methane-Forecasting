from po_valley_methane_forecasting.paths import (
    find_project_root,
)

from po_valley_methane_forecasting.rag.ingestion import (
    chunk_corpus,
)

from po_valley_methane_forecasting.rag.embedding import (
    load_embedding_model,
    embed_chunks,
)

from po_valley_methane_forecasting.rag.reranking import (
    load_reranker,
)

from po_valley_methane_forecasting.rag.retrieval import (
    retrieve_and_rerank,
)

from po_valley_methane_forecasting.rag.generation import (
    build_generation_prompt,
    generate_answer,
)


def answer_query(
    query: str,
    candidate_k: int = 50,
    top_k: int = 5,
    min_score: float = 0.35,
    max_per_source: int = 2,
    return_details: bool = False,
):
    """
    Generate an answer to a query using the complete local RAG pipeline.

    The function performs semantic retrieval, Cross-Encoder reranking,
    final chunk selection, prompt construction, and answer generation
    with Ollama.

    Parameters
    ----------
    query : str
        User question.
    candidate_k : int, default=50
        Number of chunks passed from retrieval to the reranker.
    top_k : int, default=5
        Maximum number of chunks used for answer generation.
    min_score : float, default=0.35
        Minimum similarity score for retrieval candidates.
    max_per_source : int, default=2
        Maximum number of final chunks from the same source.
    return_details : bool, default=False
        If True, also return retrieved sources and generation metrics.
    """

    project_root = find_project_root()

    documents_dir = (
        project_root
        / "rag"
        / "documents"
    )

    chunks = chunk_corpus(
        documents_dir,
        max_chars=1000,
        overlap_sentences=1,
    )

    embedding_model = load_embedding_model()

    corpus_embeddings = embed_chunks(
        chunks=chunks,
        model=embedding_model,
    )

    reranker = load_reranker()

    retrieved_chunks = retrieve_and_rerank(
        query=query,
        embedding_model=embedding_model,
        reranker=reranker,
        corpus_embeddings=corpus_embeddings,
        chunks=chunks,
        candidate_k=candidate_k,
        top_k=top_k,
        min_score=min_score,
        max_per_source=max_per_source,
    )

    prompt = build_generation_prompt(
        query,
        retrieved_chunks,
    )

    answer, metrics = generate_answer(
        prompt,
    )

    if return_details:
        return {
            "answer": answer,
            "sources": retrieved_chunks,
            "metrics": metrics,
        }

    return answer