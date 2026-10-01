import os
import asyncio
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.agent import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.cerebras import Cerebras
from tools import get_our_services, notify_sales_team, scrape_prospect_website

load_dotenv(override=True)
# Cerebras exposes an OpenAI-compatible tools API, but its LlamaIndex adapter
# defaults this capability flag to False. FunctionAgent requires it to be set.
Settings.llm = Cerebras(
    model="gpt-oss-120b",
    api_key=os.getenv("CEREBRAS_API_KEY"),
    is_function_calling_model=True,
)


# 1. Wrap the imported tools inside LlamaIndex FunctionTools
our_services_tool = FunctionTool.from_defaults(fn=get_our_services)
scrape_tool = FunctionTool.from_defaults(fn=scrape_prospect_website)
email_tool = FunctionTool.from_defaults(fn=notify_sales_team)

# 2. Initialize the agent

system_prompt = """
You are an expert AI Lead Qualification Agent for our AI Engineering & Software Services company. 
Your goal is to evaluate if a user's business can benefit from our specific services.

Follow this strict step-by-step logic loop:
1. If the user asks how we can help their business, you MUST check if you know their website or what their business does. 
   If you do not have enough context, politely ask them for their website URL and their contact email.
2. Once the user provides the website URL, call `scrape_prospect_website` to extract their company information.
3. Call `get_our_services` with the prospect's needs to search for relevant services.
4. Compare the prospect's pain points against our services. 
5. If there is a clear match, call `notify_sales_team` immediately with a concise subject and a body containing the lead's email, company information, and matched services. Then inform the user that a human engineer will reach out to them.
6. If there is no clear match, politely inform the user about the misalignment and provide alternative high-level suggestions.
"""

# FunctionCallingAgentWorker and AgentRunner were removed in LlamaIndex 0.14.
# FunctionAgent is the replacement workflow-based function-calling agent.
agent = FunctionAgent(
    tools=[our_services_tool, scrape_tool, email_tool],
    llm=Settings.llm,
    system_prompt=system_prompt,
    verbose=True
)

# 3. Executing the Production Loop

# async def main() -> None:
#     """Run two turns while keeping the agent's memory in one workflow context."""
#     context = Context(agent)

#     print(f"starting context: {context}")
    
#     # --- Turn 1: User asks a broad question ---
#     print("--- User Turn 1 ---")
#     first_response = await agent.run(
#         user_msg="How can you improve my business?", ctx=context
#     )
#     print(f"Agent Response: {first_response}\n")

#     # --- Turn 2: User provides the requested details ---
#     print("--- User Turn 2 ---")
#     final_response = await agent.run(
#         user_msg=(
#             "My website is https://gate-re.com and my email is "
#             "gatere@gate-re.com"
#         ),
#         ctx=context,
#     )
#     print(f"\nFinal Agent Response to User: {final_response}")


# if __name__ == "__main__":
#     asyncio.run(main())

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
        print(f"Agent: {response}\n")
if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 
