import os
from dotenv import load_dotenv
load_dotenv(override=True)
from llama_index.core import SimpleDirectoryReader,VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.llms.cerebras import Cerebras
from llama_index.core.tools import QueryEngineTool
# from llama_index.core.agent.function_calling import FunctionCallingAgent




# openai_api_key = os.getenv("OPENAI_API_KEY")
cerebras_api_key = os.getenv("CEREBRAS_API_KEY")
# # os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["CEREBRAS_API_KEY"] = cerebras_api_key
Settings.llm = Cerebras(model="gpt-oss-120b", api_key=cerebras_api_key)

# documents = SimpleDirectoryReader("data").load_data()
# index = VectorStoreIndex.from_documents(documents)

PERSIST_DIR = "./storage"

if not os.path.exists(PERSIST_DIR):
    # First time: load data and create index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # Save it to disk
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # Subsequent times: load directly from the local folder instantly
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# query_engine = index.as_query_engine()
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
# response = query_engine.query("How can agentrixx help my Software company to improve its employee productivity")
response = chat_engine.chat("How can agentrixx help my Software company to improve its employee productivity")
print(response)


# curriculum_tool = QueryEngineTool.from_defaults(
#     query_engine=curriculum_index.as_query_engine(),
#     description="Useful for looking up official primary school learning standards."
# )
