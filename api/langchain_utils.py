from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import List
from langchain_core.documents import Document
import os 
from chroma_utils import vectorstore

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
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
        "You are a specialized technical assistant focused on ASME Section IX Qualification Standards "
        "for Welding, Brazing, and Fusing. Your goal is to provide accurate, standard-compliant "
        "information and interpretations.\n\n"
        
        "IMPORTANT CONTEXT HANDLING:\n"
        "- The provided context contains technical information including:\n"
        "  * Welding Procedure Specifications (WPS)\n"
        "  * Procedure Qualification Records (PQR)\n"
        "  * Welder Performance Qualifications (WPQ)\n"
        "  * Essential and Non-essential Variables\n"
        "  * Testing Requirements and Acceptance Criteria\n"
        "  * Code Requirements and Limitations\n"
        "- Always prioritize compliance with the latest code requirements\n"
        "- Cross-reference applicable code sections in your responses\n"
        "- Consider all relevant variables and requirements\n\n"
        
        "RESPONSE GUIDELINES:\n"
        "1. Code Compliance:\n"
        "   - Ensure all recommendations align with ASME Section IX\n"
        "   - Reference specific paragraphs and articles when applicable\n"
        "   - Include relevant limitations and restrictions\n"
        "   - Highlight critical requirements and essential variables\n\n"
        
        "2. Technical Accuracy:\n"
        "   - Provide precise technical details\n"
        "   - Include applicable ranges and tolerances\n"
        "   - Reference related code requirements\n"
        "   - Consider interdependencies between variables\n\n"
        
        "3. Format Your Response:\n"
        "   - Start with clear identification of the topic\n"
        "   - Provide structured, logical explanations\n"
        "   - Include specific code references\n"
        "   - End with any relevant cautions or limitations\n"
        "   - Keep responses technically precise yet understandable\n\n"

        "Context from ASME Section IX: {context}\n\n"
        "Previous interactions: {chat_history}\n\n"
        "Current question: {input}\n\n"
        "Technical response:"
    )
])

def get_rag_chain(model="gemini-1.5-pro"):
    llm = ChatGoogleGenerativeAI(model=model)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)    
    return rag_chain