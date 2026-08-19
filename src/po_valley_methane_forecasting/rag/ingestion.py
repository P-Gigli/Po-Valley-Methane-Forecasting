import re

from pathlib import Path
from pypdf import PdfReader
from collections import Counter
from math import ceil


def extract_pdf_pages(
    pdf_path: Path,
) -> list[dict]:
    """
    Extract text page by page from a PDF document.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        text = page.extract_text()

        if text and text.strip():
            pages.append({
                "source": pdf_path.name,
                "page": page_number,
                "text": text.strip(),
            })

    return pages


REFERENCES_HEADING = re.compile(
    r"^\s*(references(?:\s+and\s+notes)?|bibliography)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


def remove_references_pages(
    pages: list[dict],
) -> list[dict]:
    """
    Remove the references/bibliography section
    from an extracted PDF document.
    """

    first_page_to_check = len(pages) // 2

    for page_index in range(
        len(pages) - 1,
        first_page_to_check - 1,
        -1,
    ):
        page = pages[page_index]

        match = REFERENCES_HEADING.search(
            page["text"]
        )

        if match:
            text_before_references = (
                page["text"][:match.start()].strip()
            )

            trimmed_pages = pages[:page_index]

            if text_before_references:
                trimmed_page = {
                    **page,
                    "text": text_before_references,
                }

                trimmed_pages.append(
                    trimmed_page
                )

            return trimmed_pages

    return pages


SENTENCE_BOUNDARY = re.compile(
    r'(?<=[.!?])\s+(?=["“(\[]?[A-Z0-9])'
)


ABBREVIATIONS = [
    "Fig.",
    "Figs.",
    "Sect.",
    "Sects.",
    "Eq.",
    "Eqs.",
    "e.g.",
    "i.e.",
    "et al.",
    "Dr.",
    "Prof.",
    "No.",
    "pp.",
]


DOT_PLACEHOLDER = "\uE000"


def protect_abbreviations(
    text: str,
) -> str:
    protected = text

    for abbreviation in ABBREVIATIONS:
        protected = protected.replace(
            abbreviation,
            abbreviation.replace(
                ".",
                DOT_PLACEHOLDER,
            ),
        )

    return protected


def _sentence_spans(
    text: str,
) -> list[dict]:
    """
    Split text into sentences while preserving
    their character positions.
    """

    text = protect_abbreviations(text)

    sentences = []
    start = 0

    for match in SENTENCE_BOUNDARY.finditer(text):
        end = match.start()

        sentence = text[start:end].strip()

        if sentence:
            leading_spaces = (
                len(text[start:end])
                - len(text[start:end].lstrip())
            )

            sentences.append({
                "text": sentence.replace(
                    DOT_PLACEHOLDER,
                    ".",
                ),
                "start": start + leading_spaces,
                "end": end,
            })

        start = match.end()

    # Final sentence.
    final_sentence = text[start:].strip()

    if final_sentence:
        leading_spaces = (
            len(text[start:])
            - len(text[start:].lstrip())
        )

        sentences.append({
            "text": final_sentence.replace(
                DOT_PLACEHOLDER,
                ".",
            ),
            "start": start + leading_spaces,
            "end": len(text),
        })

    return sentences


def get_edge_lines(
    text: str,
    n_lines: int = 2,
) -> tuple[list[str], list[str]]:
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    return (
        lines[:n_lines],
        lines[-n_lines:],
    )


def normalize_repeated_line(
    line: str,
) -> str:
    line = re.sub(
        r"(?<![A-Za-z])\d+(?![A-Za-z])",
        "<NUM>",
        line,
    )

    line = re.sub(
        r"\s+",
        " ",
        line,
    )

    return line.strip()


def find_repeated_edge_lines(
    pages: list[dict],
    n_lines: int = 2,
    min_fraction: float = 0.3,
) -> set[str]:
    """
    Identify normalized header/footer lines repeated
    across a substantial fraction of PDF pages.
    """

    if not 0 < min_fraction <= 1:
        raise ValueError(
            "min_fraction must be between 0 and 1."
        )

    counts = Counter()

    for page in pages:
        top_lines, bottom_lines = get_edge_lines(
            page["text"],
            n_lines=n_lines,
        )

        # A pattern should count at most once per page.
        page_patterns = {
            normalize_repeated_line(line)
            for line in top_lines + bottom_lines
        }

        counts.update(page_patterns)

    minimum_occurrences = max(
        2,
        ceil(len(pages) * min_fraction),
    )

    return {
        pattern
        for pattern, count in counts.items()
        if count >= minimum_occurrences
    }


def remove_repeated_edge_lines(
    text: str,
    repeated_patterns: set[str],
    n_lines: int = 2,
) -> str:
    """
    Remove detected repeated headers and footers
    from a page.
    """

    lines = text.splitlines()

    non_empty_indices = [
        index
        for index, line in enumerate(lines)
        if line.strip()
    ]

    edge_indices = set(
        non_empty_indices[:n_lines]
        + non_empty_indices[-n_lines:]
    )

    cleaned_lines = []

    for index, line in enumerate(lines):
        is_repeated_edge = (
            index in edge_indices
            and normalize_repeated_line(line)
            in repeated_patterns
        )

        if not is_repeated_edge:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def normalize_domain_text(
    text: str,
) -> str:

    replacements = {
        r"\bXCH\s+4\b": "XCH4",
        r"\bCH\s+4\b": "CH4",
        r"\bCO\s+2\b": "CO2",
        r"\bXCO\s+2\b": "XCO2",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(
            pattern,
            replacement,
            text,
        )

    return text


def clean_page_text(
    text: str,
) -> str:
    """
    Clean common PDF text-extraction artifacts.
    """

    # Rejoin words split by line-end hyphenation.
    text = re.sub(
        r"(\w)-\n(\w)",
        r"\1\2",
        text,
    )

    # Replace line breaks inside paragraphs with spaces.
    text = re.sub(
        r"(?<!\n)\n(?!\n)",
        " ",
        text,
    )

    # Collapse repeated spaces and tabs.
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # Reduce multiple blank lines to one paragraph break.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    text = normalize_domain_text(text)

    return text.strip()


def extract_pdf_title(
    pdf_path: Path,
) -> str:
    return (
        Path(pdf_path)
        .stem
        .replace("_", " ")
    )


def extract_markdown_title(
    md_path: Path,
) -> str:
    """
    Extract the first level-1 Markdown heading.
    """

    text = Path(md_path).read_text(
        encoding="utf-8"
    )

    match = re.search(
        r"^\s*#\s+(.+?)\s*$",
        text,
        flags=re.MULTILINE,
    )

    if match:
        return match.group(1).strip()

    # Fallback if no level-1 heading is found.
    return (
        Path(md_path)
        .stem
        .replace("_", " ")
        .strip()
    )


def extract_markdown_sections(
    md_path: Path,
) -> list[dict]:

    text = Path(md_path).read_text(
        encoding="utf-8"
    )

    heading_pattern = re.compile(
        r"^##\s+(.+?)\s*$",
        flags=re.MULTILINE,
    )

    matches = list(
        heading_pattern.finditer(text)
    )

    sections = []

    for i, match in enumerate(matches):
        start = match.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        section_text = text[start:end].strip()

        if section_text:
            sections.append({
                "section": match.group(1).strip(),
                "text": section_text,
            })

    return sections


def chunk_text(
    text: str,
    max_chars: int = 1000,
    overlap_sentences: int = 1,
) -> list[dict]:
    """
    Split text into chunks made of complete sentences.

    Chunks have approximately max_chars characters.
    Consecutive chunks may overlap by a number of
    complete sentences.
    """

    if max_chars <= 0:
        raise ValueError(
            "max_chars must be positive."
        )

    if overlap_sentences < 0:
        raise ValueError(
            "overlap_sentences cannot be negative."
        )

    sentences = _sentence_spans(text)

    chunks = []
    start_index = 0

    while start_index < len(sentences):
        end_index = start_index
        current_length = 0

        while end_index < len(sentences):
            sentence = sentences[end_index]["text"]

            additional_length = len(sentence)

            if end_index > start_index:
                additional_length += 1

            if (
                end_index > start_index
                and current_length + additional_length
                > max_chars
            ):
                break

            current_length += additional_length
            end_index += 1

        selected_sentences = sentences[
            start_index:end_index
        ]

        chunk_text_value = " ".join(
            sentence["text"]
            for sentence in selected_sentences
        )

        chunks.append({
            "text": chunk_text_value,
            "start_char": selected_sentences[0]["start"],
            "end_char": selected_sentences[-1]["end"],
        })

        if end_index >= len(sentences):
            break

        start_index = max(
            start_index + 1,
            end_index - overlap_sentences,
        )

    return chunks


def chunk_pdf(
    pdf_path: Path,
    max_chars: int = 1000,
    overlap_sentences: int = 1,
) -> list[dict]:
    """
    Extract and chunk a PDF while preserving
    page provenance.
    """

    pages = extract_pdf_pages(pdf_path)

    pages = remove_references_pages(pages)

    document_title = extract_pdf_title(
        pdf_path
    )

    repeated_edge_patterns = (
        find_repeated_edge_lines(pages)
    )

    document_parts = []
    page_spans = []

    current_position = 0

    for page in pages:

        page_text = remove_repeated_edge_lines(
            text=page["text"],
            repeated_patterns=repeated_edge_patterns,
        )

        cleaned_text = clean_page_text(
            page_text
        )

        if not cleaned_text:
            continue

        # Separate consecutive pages by one space,
        # without forcing a sentence boundary.
        if document_parts:
            document_parts.append(" ")
            current_position += 1

        page_start = current_position

        document_parts.append(cleaned_text)

        current_position += len(cleaned_text)

        page_end = current_position

        page_spans.append({
            "page": page["page"],
            "start": page_start,
            "end": page_end,
        })

    document_text = "".join(document_parts)

    raw_chunks = chunk_text(
        text=document_text,
        max_chars=max_chars,
        overlap_sentences=overlap_sentences,
    )

    chunks = []

    for chunk_index, chunk in enumerate(
        raw_chunks
    ):
        pages_in_chunk = [
            page_info["page"]
            for page_info in page_spans
            if (
                page_info["start"]
                < chunk["end_char"]
                and page_info["end"]
                > chunk["start_char"]
            )
        ]

        chunks.append({
            "source": Path(pdf_path).name,
            "source_type": "pdf",
            "document_title": document_title,
            "page_start": min(pages_in_chunk),
            "page_end": max(pages_in_chunk),
            "chunk_index": chunk_index,
            "text": chunk["text"],
            "embedding_text": (
                f"Document: {document_title}. "
                f"{chunk['text']}"
            ),
        })

    return chunks


def chunk_markdown(
    md_path: Path,
    max_chars: int = 1000,
    overlap_sentences: int = 1,
) -> list[dict]:
    """
    Extract and chunk a markdown file keeping track of
    section provenance.
    """

    document_title = extract_markdown_title(
        md_path
    )

    sections = extract_markdown_sections(
        md_path
    )

    chunks = []
    chunk_index = 0

    for section in sections:
        raw_chunks = chunk_text(
            text=section["text"],
            max_chars=max_chars,
            overlap_sentences=overlap_sentences,
        )

        for chunk in raw_chunks:
            text = chunk["text"]

            chunks.append({
                "source": Path(md_path).name,
                "source_type": "markdown",
                "document_title": document_title,
                "section": section["section"],
                "chunk_index": chunk_index,
                "text": text,
                "embedding_text": (
                    f"Document: {document_title}. "
                    f"Section: {section['section']}. "
                    f"{text}"
                ),
            })

            chunk_index += 1

    return chunks


def chunk_corpus(
    documents_dir: Path,
    max_chars: int = 1000,
    overlap_sentences: int = 1,
) -> list[dict]:
    """
    Chunk all PDF documents and markdown files contained recursively
    in a directory.
    """

    documents_dir = Path(documents_dir)

    if not documents_dir.exists():
        raise FileNotFoundError(
            f"Documents directory not found: "
            f"{documents_dir}"
        )

    pdf_paths = sorted(
        documents_dir.rglob("*.pdf")
    )

    md_paths = sorted(
        documents_dir.rglob("*.md")
    )

    document_paths = [
        *pdf_paths,
        *md_paths,
    ]

    if not document_paths:
        raise ValueError(
            f"No PDF or Markdown files found in {documents_dir}"
        )

    all_chunks = []

    for document_path in document_paths:

        suffix = document_path.suffix.lower()

        if suffix == ".pdf":
            document_chunks = chunk_pdf(
                document_path,
                max_chars=max_chars,
                overlap_sentences=overlap_sentences,
            )

        elif suffix == ".md":
            document_chunks = chunk_markdown(
                document_path,
                max_chars=max_chars,
                overlap_sentences=overlap_sentences,
            )

        else:
            continue

        category = (
            document_path.parent
            .relative_to(documents_dir)
            .as_posix()
        )

        for chunk in document_chunks:
            chunk["category"] = category
            chunk["chunk_id"] = len(all_chunks)
            all_chunks.append(chunk)

    return all_chunks