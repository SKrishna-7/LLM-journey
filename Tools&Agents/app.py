import streamlit as st
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper,DuckDuckGoSearchAPIWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler

import os
from dotenv import load_dotenv

load_dotenv()
# os.environ['HUGGING_FACE_TOKEN']=os.getenv('HUGGING_FACE_TOKEN')
# embeddings=HuggingFaceEmbeddings(model_name='')

groq_api_key=os.getenv("GROQ_API")


##Creating Tools

arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=wiki_wrapper)

search=DuckDuckGoSearchRun(name='search')

st.title("Langchain - Chat with Search")
##Sidebar
st.sidebar.title("Settings")
# groq_api_key=st.sidebar.text_input("Enter Your Gorq Api Key",type='password')


if 'messages' not in st.session_state:
    st.session_state['messages']=[
        {
            'role':'assistant',
            'content':"Hi , I'm a chatbot ,who can search the web , How can i help"
        }
    ]
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if propmt:=st.chat_input(placeholder="What is Tools and Agents"):
    st.session_state.messages.append({
        'role':'user',
        'content':propmt
    })
    st.chat_message('user').write(propmt)
    model=ChatGroq(groq_api_key=groq_api_key,model='Llama3-8b-8192',streaming=True)

    tools=[search,arxiv,wiki]

    search_agent=initialize_agent(tools,model,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,)

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])

        st.session_state.messages.append({
            'role':'assistant',
            'content':response
            })
        print(response)
        st.write(response)
