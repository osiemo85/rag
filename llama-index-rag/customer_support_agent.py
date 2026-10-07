import os
import asyncio
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.agent import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.cerebras import Cerebras
from tools import search_services_and_projects, notify_sales_team, scrape_prospect_website, send_email_to_client

load_dotenv(override=True)
# Cerebras exposes an OpenAI-compatible tools API, but its LlamaIndex adapter
Settings.llm = Cerebras(
    model="gpt-oss-120b",
    api_key=os.getenv("CEREBRAS_API_KEY"),
    is_function_calling_model=True,
)

# 1. Wrap the imported tools inside LlamaIndex FunctionTools
our_services_tool = FunctionTool.from_defaults(async_fn=search_services_and_projects)
scrape_tool = FunctionTool.from_defaults(fn=scrape_prospect_website)
email_tool = FunctionTool.from_defaults(fn=notify_sales_team)
client_email_tool = FunctionTool.from_defaults(fn=send_email_to_client)

# 2. Initialize the agent

system_prompt = """
You are a Customer Support Agent for Agentrixx Web. Your goal is to determine whether a prospect’s business can benefit from Agentrixx’s services.

* If the user asks how Agentrixx can help, check whether their website is known. If not, ask for their website URL.
* Once the website is provided, research the company to understand its business, industry, and needs.
* After reading the website, call search_services_and_projects once to get the verified Agentrixx services and project examples. Use that source to choose the most relevant matches.
* Present the potential matches to the user and ask whether the assessment is correct.
* If the user disagrees, use their feedback to refine the search until there is a clear match or no suitable match.
* Base all recommendations strictly on Agentrixx’s actual services and projects. Never guess, invent, or suggest services outside the company’s offerings.
* If there is a clear match, confirm the prospect’s email address before sending any emails.
* Notify the sales team with the prospect’s email, company information, needs, and relevant matched services.
* Send the prospect a concise, personalized email explaining the relevant services and informing them that an engineer will follow up.
* Report whether each email was successfully sent. Never claim an email was sent if it failed.
* If there is no suitable match, politely explain the misalignment and provide only relevant alternatives supported by Agentrixx’s actual offerings.
* Use relevant project examples from the retrieved service document, especially HomeHunt Kenya or Payment Manager for real estate prospects.

* If user wants help for services do not ask follow up questions, be proactive and provide immediate information on our services and actual projects that can relate to their needs.
Asking follow questions makes clients angry, give them something concrete and wait if it have feedback.
"""

# FunctionAgent is the replacement workflow-based function-calling agent.
agent = FunctionAgent(
    tools=[our_services_tool, scrape_tool, email_tool, client_email_tool],
    llm=Settings.llm,
    system_prompt=system_prompt,
    verbose=True
)

async def main() -> None:
    """Run an interactive terminal chat session while maintaining agent memory."""
    context = Context(agent)

    print(f"Starting context: {context}")
    print("\n🤖 Agent initialized! Type 'exit' or 'quit' to end the conversation.\n")

    while True:
        # 1. Capture interactive input from the user in the terminal
        user_input = input("You: ")

        # 2. Check for a breaking condition to exit the loop cleanly
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # 3. Skip empty entries
        if not user_input.strip():
            continue

        print("\nThinking...")

        # 4. Await the agent's response using the live user input
        response = await agent.run(
            user_msg=user_input,
            ctx=context
        )

        # 5. Display the response back to the user
        print(f"Agent Response: {response}\n")
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
