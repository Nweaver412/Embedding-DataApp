import os
import logging
import streamlit as st
import pandas as pd
import numpy as np

from keboola.component import CommonInterface
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore

from langchain.callbacks import StreamlitCallbackHandler

from typing import List
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

ci = CommonInterface()
input_table = ci.get_input_table_definition_by_name('embed_flow.csv')
csv_path = input_table.full_path

df = pd.read_csv(csv_path)

if "messages" not in st.session_state:
    st.session_state.messages = []
    ai_intro = "Hello, I'm Kai, your AI Assistant. I'm here to help you with your questions. What can I do for you?"
    st.session_state.messages.append({"role": "assistant", "content": ai_intro})

class UseEmbeddings(BaseEmbedding):
    def __init__(self, embeddings_dict):
        super().__init__()
        self.embeddings_dict = embeddings_dict

    def embed_query(self, query: str) -> List[float]:
        return np.zeros(len(next(iter(self.embeddings_dict.values())))).tolist()

    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        return [self.embeddings_dict[doc.doc_id].tolist() for doc in documents]

    async def aget_query_embedding(self, query: str) -> List[float]:
        return self.embed_query(query)

    async def aget_text_embedding(self, text: str) -> List[float]:
        return self.embed_query(text)  # Using embed_query as a placeholder

    def get_query_embedding(self, query: str) -> List[float]:
        return self.embed_query(query)

    def get_text_embedding(self, text: str) -> List[float]:
        return self.embed_query(text)  # Using embed_query as a placeholde

documents = []
embeddings_dict = {}

for i, row in df.iterrows():
    doc_id = str(i)
    documents.append(Document(doc_id=doc_id, text=row['text']))
    embeddings_dict[doc_id] = np.array(eval(row['embedding']))

embed_model = UseEmbeddings(embeddings_dict)

vector_store = LanceDBVectorStore()

for doc_id, embedding in embeddings_dict.items():
    vector_store.add_vector(doc_id, embedding)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context, 
    embed_model=embed_model
)

retriever = VectorIndexRetriever(index=index, similarity_top_k=5)

query_engine = RetrieverQueryEngine.from_args(retriever, node_postprocessors=[])

st.title("Kai - Your AI Assistant")

user_input = st.chat_input("Ask a question")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("Kai"):
        st.markdown("Thinking...")
    st_callback = StreamlitCallbackHandler(st.container())
    response = query_engine.query(user_input)

    st.session_state.messages.append({"role": "assistant", "content": str(response)})
    with st.chat_message("Kai"):
        st.markdown(str(response))

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
