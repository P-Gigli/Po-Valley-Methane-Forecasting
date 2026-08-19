from ollama import chat
from textwrap import dedent

def format_source_reference(
    chunk: dict,
) -> str:
    """
    Format the source reference of a retrieved chunk
    """

    source_type = chunk.get(
        "source_type"
    )

    if source_type == "pdf":
        page_start = chunk["page_start"]
        page_end = chunk["page_end"]

        if page_start == page_end:
            return f"Pages: p. {page_start}"

        return (
            f"Pages: pp. "
            f"{page_start}-{page_end}"
        )

    if source_type == "markdown":
        return (
            f"Section: {chunk['section']}"
        )

    return ""


def build_context(
    retrieved_chunks: list[dict],
) -> str:
    """
    Build a formatted textual context from
    retrieved chunks.
    """

    context_parts = []

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        source_reference = (
            format_source_reference(chunk)
        )

        context_part = (
            f"=== SOURCE [SRC_{i}] ===\n"
            f"Source: {chunk['source']}\n"
            f"{source_reference}\n"
            f"Text:\n{chunk['text']}\n"
            f"=== END SOURCE [SRC_{i}] ==="
        )

        context_parts.append(
            context_part
        )

    return "\n\n".join(context_parts)


def build_generation_prompt(
    query: str,
    retrieved_chunks: list[dict],
) -> str:
    """
    Build a grounded prompt for scientific
    question answering.
    """

    context = build_context(
        retrieved_chunks
    )

    prompt = dedent(f"""
        You are a scientific research assistant.

        Answer the QUESTION rigorously using only the information
        contained in the provided SOURCES.

        Compare all relevant sources before answering.
        Source order does not indicate importance.
        Combine evidence from multiple sources when useful.

        If the question asks about the relationship between two
        scientific quantities or concepts:
        1. Identify how the relevant quantities are explicitly
        defined in the sources.
        2. Distinguish direct measurements from quantities that are
        inferred, estimated, or derived from those measurements.
        3. Use those distinctions to answer the question.

        If a conclusion follows directly from a technical definition
        given in the sources, state that definition before drawing
        the conclusion.

        If the sources are insufficient to answer the question,
        say so explicitly. Do not introduce unsupported scientific
        claims or explanations.

        Each source in the SOURCES section has a unique identifier
        of the form [SRC_1], [SRC_2], [SRC_3], etc.
        Bracketed references appearing inside the source text are part
        of the documents and are not source identifiers.

        Write the answer in 2-3 paragraphs.
        At the end of every paragraph, append one or more exact
        source identifiers that support that paragraph.
        Use only the identifiers provided in the SOURCES section.
        Cite only sources that support the paragraph.

        QUESTION:
        {query}

        SOURCES:
        {context}

        ANSWER:
    """).strip()

    return prompt


DEFAULT_GENERATION_MODEL = "qwen3:1.7b"


def generate_answer(
    prompt: str,
    model: str = DEFAULT_GENERATION_MODEL,
) -> tuple[str, dict]:
    """
    Generate an answer locally using Ollama
    and return basic generation metrics.
    """

    response = chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        think=False,
        options={
            "temperature": 0,
            "seed": 42,
        },
    )

    metrics = {
        "prompt_tokens": response.prompt_eval_count,
        "output_tokens": response.eval_count,
        "prompt_eval_duration": response.prompt_eval_duration,
        "generation_duration": response.eval_duration,
        "total_duration": response.total_duration,
    }

    return response.message.content, metrics