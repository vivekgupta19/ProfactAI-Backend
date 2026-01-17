from typing import List, Dict, Iterator

from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_EMBEDDING_MODEL,
    validate_config,
)

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        validate_config()
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def embed_texts(texts: List[str]) -> List[List[float]]:
    client = get_client()
    response = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=texts)
    # Ensure order is preserved
    return [item.embedding for item in response.data]


def chat_with_context(question: str, context_chunks: List[Dict]) -> str:
    """Call ChatGPT with optional RAG context."""

    client = get_client()

    if context_chunks:
        context_str = "\n\n".join(
            [
                f"[Source {i + 1}] (file={chunk['metadata'].get('file')}, sheet={chunk['metadata'].get('sheet')}, row={chunk['metadata'].get('row_index')})\n{chunk['text']}"
                for i, chunk in enumerate(context_chunks)
            ]
        )
    else:
        context_str = "No external context provided. Answer from your general knowledge."

    system_prompt = (
        "You are an AI assistant answering questions about company-related data that "
        "primarily comes from Excel spreadsheets (multiple files, sheets, and different columns). "
        "Use the provided context faithfully. If the answer is not in the context, say you don't know "
        "instead of hallucinating. Be concise but clear."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Context:\n{context_str}\n\nQuestion: {question}",
        },
    ]

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
    )

    return completion.choices[0].message.content.strip()


def chat_with_context_stream(question: str, context_chunks: List[Dict]) -> Iterator[str]:
    """Stream a ChatGPT answer token-by-token (or chunk-by-chunk).

    Yields small text fragments that can be forwarded directly to the client
    as a streaming HTTP response.
    """

    client = get_client()

    if context_chunks:
        context_str = "\n\n".join(
            [
                f"[Source {i + 1}] (file={chunk['metadata'].get('file')}, sheet={chunk['metadata'].get('sheet')}, row={chunk['metadata'].get('row_index')})\n{chunk['text']}"
                for i, chunk in enumerate(context_chunks)
            ]
        )
    else:
        context_str = "No external context provided. Answer from your general knowledge."

    system_prompt = (
        "You are an AI assistant answering questions about company-related data that "
        "primarily comes from Excel spreadsheets (multiple files, sheets, and different columns). "
        "Use the provided context faithfully. If the answer is not in the context, say you don't know "
        "instead of hallucinating. Be concise but clear."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Context:\n{context_str}\n\nQuestion: {question}",
        },
    ]

    stream = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
        stream=True,
    )

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if not delta or not getattr(delta, "content", None):
            continue
        yield delta.content
