from qdrant_client import QdrantClient
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import SimpleDirectoryReader, SummaryIndex, VectorStoreIndex, Settings, StorageContext

# Read OpenAI API key from a secret file
f = open("/run/secrets/openai_key", "r")
openai_api_key: str = f.read()
f.close()

# Configure LlamaIndex settings with OpenAI's GPT-4o-mini model
Settings.llm = OpenAI(api_key=openai_api_key, model="gpt-4o-mini")
Settings.chunk_size = 1000
Settings.chunk_overlap = 50
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=openai_api_key)

# Initialize Qdrant (vector database) client
qc = QdrantClient("http://vector_db:6333", port=6333, grpc_port=6333)

# Define a vector store using Qdrant
vector_store = QdrantVectorStore(collection_name="llamaindex-docs", client=qc, enable_hybrid=True)

# Load documents from a directory recursively
docs = SimpleDirectoryReader(input_dir="/app/data", recursive=True).load_data()

# Create a storage context for indexing
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Build a vector store index from documents
v_index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

# Build a summary index from documents
s_index = SummaryIndex.from_documents(docs)

# Persist the summary index to a storage directory
s_index.storage_context.persist(persist_dir="/app/summary")