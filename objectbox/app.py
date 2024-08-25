import os               #or operating system
import time 
import streamlit as st 
from langchain_groq import ChatGroq                 #to interact with opensource LLM
from langchain_openai import OpenAIEmbeddings               #for connecting OpenAI and Langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter              #for dividing dcouments into chunks
from langchain.chains.combine_documents import create_stuff_documents_chain             #for combining documents 
from langchain_core.prompts import ChatPromptTemplate                   #for creating own custom prompt
from langchain.chains import create_retrieval_chain                 #for retrieving 
from langchain_objectbox.vectorstores import ObjectBox                  #object box data base
from langchain_community.document_loaders import PyPDFDirectoryLoader                  #for reading pdfs

from dotenv import load_dotenv
load_dotenv()


##load the Groq And OpenAI Api Key
os.environ['OPEN_API_KEY'] = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')



#creating model
st.title("Objectbox VectorstoreDB With Llama3 Demo")

llm = ChatGroq(groq_api_key = groq_api_key,
             model_name = "Llama3-8b-8192")


###creating prompt 
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}

    """

)

 ###vector Enbedding and Objectbox Vectorstore db

def vector_embedding():

    #saving sessions
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./credit_card")       ##data Ingestion
        st.session_state.docs = st.session_state.loader.load()              ##documents Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vectors = ObjectBox.from_documents(st.session_state.final_documents, st.session_state.embeddings, embedding_dimensions = 768)


#objectbox Vectorstore db
input_prompt = st.text_input("Enter Your Question From Documents")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("ObjectBox Database is ready")


if input_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()

    ##invoking LangChain
    response = retrieval_chain.invoke({'input':input_prompt})

    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    #with a streamlit expander
    with st.expander("Document Similarity Search"):
        #find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")



#--> streamlit run app.py 