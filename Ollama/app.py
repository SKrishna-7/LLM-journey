import streamlit as st
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['LANGCHAIN_KEY']=os.getenv("LANGCHAIN_KEY")
os.environ['LANGCHAIN_PROJECT']=os.getenv("LANGCHAIN_PROJECT")
os.environ['LANGCHAIN_TRACKING_V2']='true'

model=Ollama(model='gemma2:2b')
# model=Ollama(model='deepseek-r1:latest')

prompt=ChatPromptTemplate.from_messages(
    [
        ("system",'You are a helpful assistant. Please respond to the question asked'),
        ('user','Question:{question}')

    ]
)

st.title("Deepseek demo with Langchain")

input_text=st.text_input("What question you have in mind?")
output_parser=StrOutputParser()
chain=prompt|model|output_parser
if input_text:
    res=chain.invoke({'question':input_text})
    # print(res)
    st.write(res)