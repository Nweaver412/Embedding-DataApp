import os
import logging
import streamlit as st
import pandas as pd
import ast

from keboola.component import CommonInterface
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
else:
    st.error("OPENAI_API_KEY is not set. Please check your .env file.")

ci = CommonInterface()
input_table = ci.get_input_table_definition_by_name('app-embed-lancedb.csv')
csv_path = input_table.full_path
df = pd.read_csv(csv_path)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello, I'm Kai, your AI Assistant. I'm here to help you with your questions. What can I do for you?"}
    ]

documents = [
    Document(doc_id=str(i), text=row['bodyData'], embedding=ast.literal_eval(row['embedding']))
    for i, row in df.iterrows()
]

embeddings_dict = {str(i): ast.literal_eval(row['embedding']) for i, row in df.iterrows()}
vector_store = LanceDBVectorStore.from_dict(embeddings_dict)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
query_engine = RetrieverQueryEngine.from_args(retriever, node_postprocessors=[])

st.title("Kai - Your AI Assistant")

user_input = st.chat_input("Ask a question")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    st_callback = StreamlitCallbackHandler(st.container())
    response = query_engine.query(user_input)
    response_text = str(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})

    with st.chat_message("assistant"):
        st.markdown(response_text)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])