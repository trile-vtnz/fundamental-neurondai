import os
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

class DocumentManager:
    def __init__(self, folder="documents"):
        self.folder = folder
        self._ensure_folder_exists()

    def _ensure_folder_exists(self):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def load_documents(self):
        """Loads documents from the specified folder."""
        reader = SimpleDirectoryReader(self.folder)
        return reader.load_data()

class IndexManager:
    def __init__(self, document_manager):
        self.document_manager = document_manager
        self.index = None

    def build_index(self):
        """Builds an index from the documents."""
        docs = self.document_manager.load_documents()
        self.index = VectorStoreIndex.from_documents(docs)

    def get_query_engine(self):
        """Returns a query engine from the index."""
        if self.index is None:
            self.build_index()
        return self.index.as_query_engine()

class RAGDemoApp:
    def __init__(self):
        load_dotenv()
        self.document_manager = DocumentManager()
        self.index_manager = IndexManager(self.document_manager)
        self.setup_streamlit()

    def setup_streamlit(self):
        st.set_page_config(page_title="RAG Demo with LlamaIndex", page_icon=":books:")
        st.title("RAG Demo using LlamaIndex")
        st.write("This demo allows you to query documents stored in the `documents` folder.")

    def run(self):
        if "query_engine" not in st.session_state:
            # Configure the global LLM settings
            Settings.llm = OpenAI(model="gpt-4o-mini", streaming=True)
            st.session_state.query_engine = self.index_manager.get_query_engine()

        if "messages" not in st.session_state:
            st.session_state.messages = []

        self.display_chat()

    def display_chat(self):
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask question about the documents"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response = st.session_state.query_engine.query(prompt)
                st.markdown(response.response)
                st.session_state.messages.append({"role": "assistant", "content": response.response})

if __name__ == "__main__":
    app = RAGDemoApp()
    app.run()
