import streamlit as st

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


prompts=ChatPromptTemplate.from_messages(
    [
        ('system','You are a AI assistant '),
        ('human','Question: {questions}')
    ]
)

parser=StrOutputParser()

def Generate_response(question,models,temperature,max_token):

    model=Ollama(model=models)
    
    chain=prompts|model|parser

    response = chain.invoke({'questions':question})

    return response


st.title("Simple Q&A Chatbot With Ollama")


user_input=st.text_input("Ask Your Queries..")

models=st.sidebar.selectbox('Select the Ollama Model',['gemma2:2b','mistral:latest'])


temperature=st.sidebar.slider("Temperature ",min_value=0.0,max_value=1.0,value=0.7)
max_token=st.sidebar.slider("Max Tokens ",min_value=50,max_value=300,value=150)

st.sidebar.write('@Suresh Krishnan')
if user_input:
    res=Generate_response(user_input,models,temperature,max_token)
    st.write(res)