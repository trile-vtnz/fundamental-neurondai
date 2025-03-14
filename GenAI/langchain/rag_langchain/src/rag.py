from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

# Document loader
# https://python.langchain.com/docs/modules/data_connection/document_loaders/
# https://python.langchain.com/docs/integrations/document_loaders/
print("Document loader is loading documents...")
from langchain_community.document_loaders import YoutubeLoader
loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=QfAnHdEnP-k&t=1874s", language='vi')
documents = loader.load()

# Split documents with text splitter
# https://python.langchain.com/docs/modules/data_connection/document_transformers/
print("Text splitter is splitting documents...")
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=0
)
chunks = []
for document in documents:
    chunks += (text_splitter.create_documents([document.page_content]))


# Store our documents in a vector store
# https://python.langchain.com/docs/modules/data_connection/vectorstores/
# (optional) add persist_directory so we can reuse the db without re-creating it
print("Storing documents and embeddings in vector store...")
db = Chroma.from_documents(chunks, OpenAIEmbeddings())

print("Ready to ask!\n###########################################\n")

# Create a retriever with our vector store
# https://python.langchain.com/docs/modules/data_connection/retrievers/
retriever = db.as_retriever()

# Chat model with stdout streaming output
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)

# Create a prompt template https://python.langchain.com/docs/modules/model_io/prompts/
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, HumanMessagePromptTemplate
prompt = ChatPromptTemplate(
    input_variables=['context', 'question'],
    messages=[
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['context', 'question'],
                template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"
                )
            )
    ]
)

# list of questions to ask
questions = [
    "Ai là diễn giả trong buổi này ?",
    "Chương trình này có tên là gì",
    "RAG hoạt động như thế nào"
]

# put everything together
for question in questions:
    print(f"Question: {question}\n")
    # search for similar documents
    docs = retriever.invoke(question)
    # create context merging docs together
    context = "\n\n".join(doc.page_content for doc in docs)
    # get valorized prompt from template
    prompt_val = prompt.invoke({"context": context, "question": question})
    # get response from llm
    result = llm.invoke(prompt_val.to_messages()).content
    print(f"Answer: {result}")
    print("\n#########################################\n")