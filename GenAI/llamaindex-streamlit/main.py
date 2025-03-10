import os
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Define the folder where documents are stored
documents_folder = "documents"

def load_documents():
    """Loads documents from the specified folder."""
    if not os.path.exists(documents_folder):
        os.makedirs(documents_folder)
    reader = SimpleDirectoryReader(documents_folder)
    return reader.load_data()

def build_index():
    """Builds an index from the documents."""
    docs = load_documents()
    index = VectorStoreIndex.from_documents(docs)
    return index

def main():
    st.set_page_config(page_title="RAG Demo with LlamaIndex", page_icon=":books:")
    st.title("RAG Demo using LlamaIndex")

    st.write("This demo allows you to query documents stored in the `documents` folder.")

    if "index" not in st.session_state:
        # Configure the global LLM settings
        Settings.llm = OpenAI(model="gpt-3.5-turbo", streaming=True)
        st.session_state.index = build_index()

    query_engine = st.session_state.index.as_query_engine()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up !"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = query_engine.query(prompt)
            st.markdown(response.response)
            st.session_state.messages.append({"role": "assistant", "content": response.response})

if __name__ == "__main__":
    main()
