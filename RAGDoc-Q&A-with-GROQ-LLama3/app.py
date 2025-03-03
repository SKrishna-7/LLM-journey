import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv

load_dotenv()

os.environ['GROQ_API']=os.getenv("GROQ_API")

groq_api_key=os.getenv("GROQ_API")

llm_model=ChatGroq(groq_api_key=groq_api_key,model_name='llama-3.3-70b-versatile')

prompt_temp=ChatPromptTemplate.from_template(
    """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question
        <context>
            {context}
        </context>

        Question:{input}
    """
)


def create_vectors_embeddings():
    
    if "Vectors" not in st.session_state:
        st.session_state.embeddings=OllamaEmbeddings(model='mistral:latests')
        st.session_state.loader=PyPDFDirectoryLoader('Pdf')
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_doc=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_doc,st.session_state.embeddings)


user_prompt=st.text_input("Enter Your Query From Document ")

if st.button("Document Embedding"):
    create_vectors_embeddings()

    st.write("Vector Database is Ready..")

import time


if user_prompt:
    
    document_chain=create_stuff_documents_chain(llm_model,prompt_temp)
    retriever=st.session_state.vectors.as_retriever()

    retriever_chain=create_retrieval_chain(retriever,document_chain)

    start=time.process_time()
    response=retriever.invoke(
        {'input':user_prompt}
    )

    print(f"Response time : {time.process_time()-start}")

    st.write(response[0].page_content)

    # with st.expander("Document similarity Search "):
    #     for i,doc in enumerate(response[0].page_content):
    #         st.write(doc.page_content)
    #         st.write('-------------------------')
    # print(response[0].page_content)
