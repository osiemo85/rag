import asyncio
import os
from pathlib import Path

from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel
from agents.mcp import MCPServerStdio
from dotenv import load_dotenv
from IPython.display import Markdown, display
from qdrant_client import QdrantClient

load_dotenv(override=True)

MODEL_NAME = "cerebras/gpt-oss-120b"
COLLECTION_NAME = "knowledge"
KNOWLEDGE_DIR = Path.cwd() / "knowledge"
VECTORDB_PATH = KNOWLEDGE_DIR / "vectordb"
INGEST_URLS = ["https://www.agentrixx.com"] * 3


def get_model() -> LitellmModel:
    return LitellmModel(model=MODEL_NAME, api_key=os.getenv("CEREBRAS_API_KEY"))


def get_vectorstore_params(vectordb_path: Path) -> dict:
    return {
        "command": "uvx",
        "args": ["mcp-server-qdrant"],
        "env": {
            "QDRANT_LOCAL_PATH": str(vectordb_path),
            "COLLECTION_NAME": COLLECTION_NAME,
        },
    }


def build_ingestion_instructions() -> str:
    return """You are an expert on Norbert's companywebsite, Agentrixx focus on technology solutions
You are populating your memories with information retrieved from a given website.
Use your MCP tools to retrieve the website. Extract key knowledge. Check what's already in your memories to avoid duplicates.
After you are done, reply with a brief status update and the number of memories you added.
Aim to add at least 10 unique memories, unless your existing memories are already comprehensive.
"""


def build_chat_instructions() -> str:
    return (
        "You are an expert on Norbert's company website, Agentrixx focus on technology solutions. "
        "You are assisting the user with their queries based on the knowledge stored in your memories."
    )


def create_ingestion_agent(fetch_mcp: MCPServerStdio, vectorstore_mcp: MCPServerStdio) -> Agent:
    return Agent(
        name="Ingester",
        model=get_model(),
        instructions=build_ingestion_instructions(),
        mcp_servers=[fetch_mcp, vectorstore_mcp],
    )


def create_chat_agent(vectorstore_mcp: MCPServerStdio) -> Agent:
    return Agent(
        name="ChatAgent",
        model=get_model(),
        instructions=build_chat_instructions(),
        mcp_servers=[vectorstore_mcp],
    )


async def get_content() -> None:
    KNOWLEDGE_DIR.mkdir(exist_ok=True)
    fetch_params = {"command": "uvx", "args": ["mcp-server-fetch"]}
    vectorstore_params = get_vectorstore_params(VECTORDB_PATH)

    for url in INGEST_URLS:
        async with MCPServerStdio(params=fetch_params, client_session_timeout_seconds=120) as fetch_mcp:
            async with MCPServerStdio(
                params=vectorstore_params,
                client_session_timeout_seconds=120,
            ) as vectorstore_mcp:
                agent = create_ingestion_agent(fetch_mcp, vectorstore_mcp)
                task = (
                    f"Add unique memories with information from this website: {url} and reply with "
                    "a one sentence status update including how many memories were added."
                )
                response = await Runner.run(agent, task, max_turns=50)
                display(Markdown(response.final_output))


async def read_content() -> None:
    client = QdrantClient(path=str(VECTORDB_PATH))
    info = client.get_collection(COLLECTION_NAME)
    print(f"Memories in '{COLLECTION_NAME}': {info.points_count}\n")

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=200,
        with_payload=True,
        with_vectors=False,
    )
    for index, point in enumerate(points, 1):
        document = (point.payload or {}).get("document", "")
        preview = document.replace("\n", " ")[:160]
        suffix = "..." if len(document) > 160 else ""
        print(f"{index:>3}. {preview}{suffix}")

    client.close()


async def chat(message: str) -> None:
    vectorstore_params = get_vectorstore_params(VECTORDB_PATH)
    async with MCPServerStdio(
        params=vectorstore_params,
        client_session_timeout_seconds=120,
    ) as vectorstore_mcp:
        agent = create_chat_agent(vectorstore_mcp)
        response = await Runner.run(agent, f"User message: {message}", max_turns=50)
        display(response.final_output)


if __name__ == "__main__":
    asyncio.run(chat("What 4 services does Agentrixx offer and what are the 4 steps on how Agentrixx works?"))
