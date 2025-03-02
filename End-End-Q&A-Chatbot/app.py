import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()


os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_KEY")
os.environ['LANGCHAIN_TRACKING_V2']="true"
os.environ['LANGCHAIN_PROJECT']="Q&A chatbot with Groq"

groq_api_key=os.getenv("GROQ_API")

#Prompt Template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpfull assistant.Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

def generate_response(question,api_key,llm,temperature,max_tokens):

    groq_model=ChatGroq(model=llm,groq_api_key=api_key)
    output_parser=StrOutputParser()

    chain=prompt|groq_model|output_parser

    response=chain.invoke(
        {
            "question":question
        }
    )

    return response


st.title("Q&A Chaatbot With Groq Deepseek 70B")

api_key=st.sidebar.text_input("Enter Your GROQ API Key : ",type='password')

model=st.sidebar.selectbox("Select an LLM model ",["deepseek-r1-distill-llama-70b",'gemma2-9b-it'])

temperature=st.sidebar.slider("Temperature ",min_value=0.0,max_value=1.0,value=0.7)
max_token=st.sidebar.slider("Max Tokens ",min_value=50,max_value=300,value=150)

st.write("Ask any Question..!")
user_input=st.text_input("You : ")

if user_input:  
    response=generate_response(user_input,groq_api_key,model,temperature,max_token)
    st.write(response)

else:
    st.warning("Provide a input text...!")