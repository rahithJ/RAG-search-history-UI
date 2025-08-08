import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain

import os
from dotenv import load_dotenv
load_dotenv()
# os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
# os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
# os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
# HF_TOKEN = st.secrets["HF_TOKEN"]
# LANGCHAIN_API_KEY = st.secrets["HF_TOKEN"]
# GROQ_API_KEY = st.secrets["HF_TOKEN"]
with st.sidebar:
    os.environ['LANGCHAIN_API_KEY'] = st.text_input('Enter your LANGCHAIN API KEY',type='password')
    os.environ['GROQ_API_KEY'] = st.text_input('Enter your GROQ API KEY',type='password')
    os.environ['HF_TOKEN'] = st.text_input('Enter your HUGGINGFACE API KEY',type='password')

os.environ['LANGCHAIN_PROJECT'] = 'NEW_RAG_WITH_HISTORY'
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
st.title('RAG WITH HISTORY SHOWN')
"""PDF Explorer by Rahith """
uploaded_file = st.file_uploader('Upload a PDF to Disect',type='pdf',accept_multiple_files=False)

with st.sidebar:
    session_id = st.text_input('Session Id',value='CHAT-1')

if 'store' not in st.session_state:
    st.session_state.store = {}

if 'message' not in st.session_state:
    st.session_state['message'] = [
        {'role':'Human','content':'Hey,What I can do here'},
        {'role':'assistant',
        'content':'Hi, I am here to help you with the PDF that you have uploaded, ask anything about the PDF'}
    ]

for msg in st.session_state.message:
    st.chat_message(msg['role']).write(msg['content'])

if uploaded_file:
    tempfile = f'./temp.pdf'
    document = []
    with open(tempfile,'wb') as file:
        file.write(uploaded_file.getvalue())
        file_name = uploaded_file.name

    llm = ChatGroq(model='llama-3.3-70b-versatile')
    raw_doc = PyPDFLoader(tempfile).load()
    document.extend(raw_doc)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    split_document = text_splitter.split_documents(document)
    if not raw_doc:
        st.error('NO document found, try another Document')
        st.stop()
    if not split_document:
        st.error('No document is for embedding')
        st.stop()

    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    db = FAISS.from_documents(split_document,embeddings)
    reteriver = db.as_retriever()
    history_prompt_text = (
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
    
    history_prompt = ChatPromptTemplate.from_messages([
        ("system",history_prompt_text),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ])

    history_chain = create_history_aware_retriever(llm,reteriver,history_prompt)

    qa_prompt_text = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",qa_prompt_text),
        MessagesPlaceholder('chat_history'),
        ("human","{input}")
    ])
    
    document_chain = create_stuff_documents_chain(llm,qa_prompt)
    rag_chain = create_retrieval_chain(history_chain,document_chain)

    def get_session_history(session_id) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]
    
    final_contexual_query = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key='input',
        output_messages_key='answer',
        history_messages_key='chat_history'
    )

    question = st.chat_input('Ask A Question')
    if question:
        session_history = get_session_history(session_id)
        response = final_contexual_query.invoke(
            {"input":question},
            config={'configurable':{'session_id':session_id}}
        )
        st.session_state.message.append({'role':'Human','content':question})
        st.chat_message("human").write(question)

        with st.chat_message("assistant"):
            st.session_state.message.append({'role':'assistant','content':response['answer']})
            st.write("Answer: ",response['answer'])

        # with st.sidebar:
        #     st.write("Chat: ",st.session_state.store.keys())

        #     st.write("Chat History: ",session_history)



