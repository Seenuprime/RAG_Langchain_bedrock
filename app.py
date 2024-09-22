from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import boto3
import streamlit as st 


bedrock = boto3.client(service_name='bedrock-runtime')
embedding = HuggingFaceBgeEmbeddings(model_name='all-MiniLM-L6-v2')

def vectore_store_docs():
    docs = PyPDFLoader('attention_all_youneed.pdf').load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    splitted_docs = splitter.split_documents(docs)

    faiss_db = FAISS.from_documents(
        splitted_docs,
        embedding
    )

    faiss_db.save_local("faiss_index")


def model():
    from langchain_aws import ChatBedrock

    model_name="amazon.titan-text-premier-v1:0"
    llm = ChatBedrock(model_id=model_name, model_kwargs=dict(temperature=0.5))

    return llm

prompt_template = """
Human: Use the following given context given to provide the consise answer to the question and try to summarize with the 150 words,
if you dont't find the answer in the given context then you can say i couldn't find the answer in the context.
<context>
{context}
</context>

question: {question}

answer: 
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", 'question'])

def get_response(llm, vector_store, query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
    )
    answer=qa({"query":query})
    return answer['result'] 



def main():
    st.set_page_config("Chat PDF")
    st.header("Welcome to the LLM using Langchain Bedrock")

    user_question = st.text_input("Enter your query: ")
    
    with st.sidebar:
        st.title("Menu: ")
        
        if st.button("Vectore Update"):
            with st.spinner("Processing....."):
                vectore_store_docs()
                st.success("Done")

    if st.button("Model Ouput"):
        with st.spinner("Processing....."):
            faiss_index = FAISS.load_local('faiss_index', embedding, allow_dangerous_deserialization=True)
            llm = model()

            st.write(get_response(llm, faiss_index, user_question))
            st.success("Done")

main()