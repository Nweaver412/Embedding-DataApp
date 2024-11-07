import os
import logging
import streamlit as st
import pandas as pd
import numpy as np
import ast

from keboola.component import CommonInterface
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
else:
    st.error("Not set")

ci = CommonInterface()
input_table = ci.get_input_table_definition_by_name('app-embed-lancedb.csv')
csv_path = input_table.full_path
df = pd.read_csv(csv_path)

embedding_model = OpenAIEmbeddings()

documents = [
    Document(page_content=row['bodyData'], metadata={"doc_id": str(i)})
    for i, row in df.iterrows()
]
embeddings = [ast.literal_eval(row['embedding']) for row in df.iterrows()]

vector_store = FAISS.from_documents(documents, embedding_model)

st.title("Kai - Your AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello, I'm Kai, your AI Assistant. I'm here to help you with your questions. What can I do for you?"}
    ]

user_input = st.chat_input("Ask a question")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    st_callback = StreamlitCallbackHandler(st.container())
    response_docs = vector_store.similarity_search(user_input, k=5)
    response_text = "\n".join([doc.page_content for doc in response_docs])
    st.session_state.messages.append({"role": "assistant", "content": response_text})

    with st.chat_message("assistant"):
        st.markdown(response_text)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
