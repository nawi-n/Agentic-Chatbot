from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import List
from langchain_core.documents import Document
import os 
from chroma_utils import vectorstore

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
output_parser = StrOutputParser()

os.environ["GOOGLE_API_KEY"] = "AIzaSyCE3dhFwnTOWSGHdOFv9Av0MjJVwdK7VMM"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "Travel-Chatbot"

# Modified prompts to use only human messages
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template(
        "Given the following chat history and latest question, create a standalone question that captures the full context:\n\n"
        "Chat History:\n{chat_history}\n\n"
        "Latest Question: {input}\n\n"
        "Standalone question:"
    )
])

qa_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template(
        "You are a helpful AI assistant. Use the following context to answer the question.\n\n"
        "Context: {context}\n\n"
        "Chat History:\n{chat_history}\n\n"
        "Question: {input}\n\n"
        "Helpful answer:"
    )
])

def get_rag_chain(model="gemini-1.0-pro"):
    llm = ChatGoogleGenerativeAI(model=model)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)    
    return rag_chain