import streamlit as st
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI


from dotenv import load_dotenv
load_dotenv()

st.title("RAG with Gemini-2.5-Flash")

loader = PyPDFLoader("all-of-statistics.pdf")

data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(data)

embeddings = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004')
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

retriever = vectorstore.as_retriever(search_type='similarity',search_kwargs={"k": 10})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, max_tokens=500)

query = st.chat_input('Ask me anything: ')

prompt = query

system_prompt = ("""You are a helpful AI assistant that helps people find information about courses from the provided context.
If you don't know the answer, just say that you don't know. DO NOT try to make up an answer.
Use the following pieces of context to answer the question at the end.
{context}  
Answer the question truthfully and as best as you can and keep it concise.
""")


prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}")
])




if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})
    print(response['answer'])
    st.write(response['answer'])
# rag_chain = create_retrieval_chain(retriever, question_answering_chain)




