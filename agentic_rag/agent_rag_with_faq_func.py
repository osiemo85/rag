import asyncio
import json
import os
from pathlib import Path
from time import perf_counter

from agents import Agent, Runner, function_tool
from agents.extensions.models.litellm_model import LitellmModel
from agents.mcp import MCPServerStdio
from dotenv import load_dotenv
from litellm import model_cost

load_dotenv(override=True)

MODEL_NAME = "cerebras/gpt-oss-120b"
KNOWLEDGE_DIR = Path.cwd() / "knowledge"
FAQ_PATH = KNOWLEDGE_DIR / "faq.jsonl"
VECTORDB_PATH = KNOWLEDGE_DIR / "vectordb"


def load_faqs(path: Path) -> list[dict]:
    with path.open() as file:
        return [json.loads(line) for line in file]


def build_instructions(faqs: list[dict]) -> str:
    instructions = """
# Role

You are an expert about Ed Donner and his online courses. You are answering questions about him and his courses to visitors on his website.
Use your memories and tools to help find background information to answer the question. As needed, use multiple tools at the same time to gather all relevant context.
If you don't know the answer, say so.

# Memory

Always use your qdrant-find memory tool to help find relevant information. You can make multiple queries. Make all tool calls in parallel.

# FAQ

Your faq tool contains answers to all the common questions. Below is a list of the questions with their numbers.
If the user's question is related to one of these questions, then use your faq tool to retrieve a specific answer.
Respond with the answer in its original form in markdown, as written by Ed. If the answer include hyperlinks, then keep them in markdown format.

List of questions by number:
"""

    for faq in faqs:
        instructions += f"\n{faq['faq']}. {faq['question']}"
    return instructions


def get_vectorstore_params(vectordb_path: Path) -> dict:
    return {
        "command": "uvx",
        "args": ["mcp-server-qdrant"],
        "env": {
            "QDRANT_LOCAL_PATH": str(vectordb_path),
            "COLLECTION_NAME": "knowledge",
        },
    }


model = LitellmModel(model=MODEL_NAME, api_key=os.getenv("CEREBRAS_API_KEY"))
faqs = load_faqs(FAQ_PATH)
faqs_lookup = {faq["faq"]: faq for faq in faqs}
instructions = build_instructions(faqs)


def find_faq(question_number: int) -> str:
    faq = faqs_lookup.get(question_number)
    if faq:
        return f"### Question {faq['faq']} is:\n{faq['question']}\n### Ed's answer:\n{faq['answer']}"
    return "That question number was not found in the FAQ."


@function_tool(strict_mode=False)
def faq_tool(question_number: int) -> str:
    """Use this tool to retrieve the answer to a frequently asked question by its number."""
    return find_faq(question_number)


def create_agent(instructions: str, vectorstore_mcp: MCPServerStdio) -> Agent:
    return Agent(
        name="ChatAgent",
        model=model,
        instructions=instructions,
        tools=[faq_tool],
        mcp_servers=[vectorstore_mcp],
    )


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Estimate cost using LiteLLM's pricing for the configured model."""
    pricing = model_cost.get(MODEL_NAME)
    if pricing is None:
        return 0.0

    return (
        input_tokens * pricing.get("input_cost_per_token", 0.0)
        + output_tokens * pricing.get("output_cost_per_token", 0.0)
    )


async def chat(message: str) -> None:
    vectorstore_params = get_vectorstore_params(VECTORDB_PATH)
    async with MCPServerStdio(
        params=vectorstore_params,
        client_session_timeout_seconds=120,
    ) as vectorstore_mcp:
        agent = create_agent(instructions, vectorstore_mcp)
        task = f"User message: {message}"
        started_at = perf_counter()
        response = await Runner.run(agent, task, max_turns=50)
        elapsed_seconds = perf_counter() - started_at
        usage = response.context_wrapper.usage
        estimated_cost = estimate_cost_usd(usage.input_tokens, usage.output_tokens)

        print(response.final_output)
        print(
            "\n--- Response metrics ---"
            f"\nTime: {elapsed_seconds:.2f} seconds"
            f"\nModel requests: {usage.requests}"
            f"\nInput tokens: {usage.input_tokens:,}"
            f"\nOutput tokens: {usage.output_tokens:,}"
            f"\nTotal tokens: {usage.total_tokens:,}"
            f"\nEstimated cost: ${estimated_cost:.6f} USD"
        )


if __name__ == "__main__":
    asyncio.run(chat("What is Agentrixx's tagline and what are the company's 4 core service areas?"))
