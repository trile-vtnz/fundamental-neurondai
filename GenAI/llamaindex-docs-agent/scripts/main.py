from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
import gradio as gr
from qdrant_client import QdrantClient
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.core.tools import QueryEngineTool, FunctionTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.agent import ReActAgent
from llama_index.core.llms import ChatMessage
from tavily import TavilyClient
import requests

# Load environment variables from .env file
load_dotenv()

# Initialize the Tavily search client with an API key from environment variables
tavily_client = TavilyClient(api_key=os.getenv("tavily_api_key"))

# Configure the LLM settings for LlamaIndex using OpenAI's GPT-4o-mini model
Settings.llm = OpenAI(api_key=os.getenv("openai_api_key"), model="gpt-4o-mini")
Settings.chunk_size = 1000
Settings.chunk_overlap = 50
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=os.getenv("openai_api_key"))

# Initialize Qdrant (vector database) client for vector storage
qc = QdrantClient("http://vector_db:6333", port=6333, grpc_port=6333)
vector_store = QdrantVectorStore(client=qc, collection_name="llamaindex-docs", enable_hybrid=True)

# Create a vector index using the Qdrant vector store
v_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# Load an existing storage context for document summarization
s_context = StorageContext.from_defaults(persist_dir="/app/summary")
s_index = load_index_from_storage(storage_context=s_context)

# Create query engines for both the vector index and summarization index
v_engine = v_index.as_query_engine()
s_engine = s_index.as_query_engine()

def tavily_search(query: str = Field("Query to search the web for information about LlamaIndex"))-> str:
    """Search the web for LlamaIndex-related information using Tavily API."""
    response = tavily_client.search(
        query=query,
        include_answer="basic"
    )
    return response["answer"]

# Define tools for querying different sources
list_tool = QueryEngineTool.from_defaults(
    query_engine=s_engine,
    description=(
        "Useful for summarization questions related to LlamaIndex Documentation"
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=v_engine,
    description=(
        "Useful for retrieving specific context from LlamaIndex Documentation"
    ),
)

# Create a router query engine to select between summarization and vector retrieval
router_engine = RouterQueryEngine(
    selector=PydanticSingleSelector.from_defaults(),
    query_engine_tools=[
        list_tool,
        vector_tool,
    ],
)

router_tool = QueryEngineTool.from_defaults(
    query_engine=router_engine,
    description=(
        "A query engine based on a router: the router can select either from a summarization index or from a vector index."
    ),
)

# Define a web search tool using Tavily API
web_search_tool = FunctionTool.from_defaults(
    fn=tavily_search
)

# Define chat history with system instructions
chat_history = [ChatMessage.from_str(
    "You are a useful assistant for developers. Your task is to help them by providing useful information about LlamaIndex, a popular AI framework with countless integrations. You should rely on the contextual information you can find from the documentation database and, if that is not sufficient, search the internet. The documentation retrieved from the database is sufficient when it does not only provide vague guidance, but when it actually answers the user's question: to be deemed sufficient, it would be important (but not essential) that the retrieved information contains code snippets.",
    role="system")
]

# Create an AI agent using ReAct framework, equipped with query and web search tools
agent = ReActAgent.from_tools(tools=[router_tool, web_search_tool], chat_history=chat_history, verbose=True)

# Initialize FastAPI application with ORJSON for fast JSON responses
app = FastAPI(default_response_class=ORJSONResponse)

# Define a Pydantic model for user messages
class Message(BaseModel):
    message: str = Field(description="Message from the user")

# Define a root endpoint to return a simple message
@app.get("/")
async def read_main():
    return {"message": "This is your main app"}

# Define a message processing endpoint that uses the AI agent
@app.post("/message/")
async def read_message(message: Message) -> Message:
    response = agent.chat(message.message)
    msg = Message(message=str(response))
    return msg

# Define a function for handling messages in the Gradio chat interface
def reply(message, history):
    if message == "" or message is None:
        res = "You should provide me with a message"
        r = ""
        for char in res:
            r += char
            yield r
    else:
        req = requests.post("http://localhost:8000/message", json={"message": message})
        response = req.json()
        res = response["message"]
        r = ""
        for char in res:
            r += char
            yield r

# Create a Gradio chat interface for interacting with the AI agent
io = gr.ChatInterface(
    reply,
    title="Chat with LlamaIndex docs!🦙",
    examples=[
        "How do I install LlamaIndex?",
        "What is a workflow and how can I implement it?",
        "How can I build advanced retrieval with LlamaIndex?"
    ]
)

# Mount Gradio UI into the FastAPI app
app = gr.mount_gradio_app(app, io, path="/gradio")