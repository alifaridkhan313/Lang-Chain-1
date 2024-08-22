#libraries 

from langchain_openai import ChatOpenAI
from langchain_core.promts import ChatPromptTemplate    
from langchain_core.output_parsers import StrOutputParsers
from langchain_community.llms import Ollama 
import os
import streamlit as st
from dotenv import load_dotenv

#initialize
load_dotenv()


#langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] =os.getenv("LANGCHAIN_API_KEY")

#prompt template

prompt = ChatPromptTemplate.from_message(
    [
        ("system", "You are AI helpful assistant. Please respond to the queries")
        ("user", "Question: {question}")
    ]
)

#steamlit framework 

st.title("LangChain demo with Llama2 API")
input_text = st.text_input("Search any topic you want")

#Ollama LLM (llama2)

llm = Ollama(model = "llama2")
output_parser = StrOutputParsers()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'questions': input_text}))