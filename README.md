# Agentic RAG

A small Retrieval-Augmented Generation (RAG) project that stores website content in a local Qdrant vector database and answers questions using that stored knowledge. It uses the OpenAI Agents SDK, LiteLLM, Cerebras, and MCP servers for web fetching and Qdrant access.

## Requirements

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/)
- A Cerebras API key

## Setup

Clone the project, then install the dependencies:

```bash
uv sync
```

Create a `.env` file in the project root:

```env
CEREBRAS_API_KEY=your_cerebras_api_key
```

The `.env` file and local vector database are ignored by Git.

## Ingest website content

`agentic_ingest.py` is responsible for collecting website content and saving it to `knowledge/vectordb/`.

1. Update `INGEST_URLS` in `agentic_ingest.py` with the website pages you want to add.
2. Change the last line of the file to run ingestion:

```python
asyncio.run(get_content())
```

3. Run it:

```bash
uv run python agentic_ingest.py
```

The first run may download the MCP server tools through `uvx`. The local database is created automatically.

## Ask questions

To query the stored website content, set the final line of `agentic_ingest.py` to:

```python
asyncio.run(chat("What services does Agentrixx offer?"))
```

Then run:

```bash
uv run python agentic_ingest.py
```

Replace the message with your own question.

## View stored memories

To print the stored documents, set the final line of `agentic_ingest.py` to:

```python
asyncio.run(read_content())
```

Then run:

```bash
uv run python agentic_ingest.py
```

## FAQ-enabled chat

`agent_rag_with_faq_func.py` adds a local FAQ tool from `knowledge/faq.jsonl` alongside vector search. Update its final `chat(...)` message as needed, then run:

```bash
uv run python agent_rag_with_faq_func.py
```

## Project files

- `agentic_ingest.py` — ingest websites, inspect memories, and query the local vector database.
- `agent_rag_with_faq_func.py` — query the vector database with an additional FAQ lookup tool.
- `knowledge/faq.jsonl` — FAQ source data.
- `knowledge/vectordb/` — generated local Qdrant data; do not commit it.
