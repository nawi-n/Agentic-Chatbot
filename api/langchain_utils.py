from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.tools import Tool, StructuredTool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from typing import List
from langchain_core.documents import Document
import os 
from chroma_utils import vectorstore
from ticket_utils import get_ticket_details

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
output_parser = StrOutputParser()

os.environ["GOOGLE_API_KEY"] = "AIzaSyCE3dhFwnTOWSGHdOFv9Av0MjJVwdK7VMM"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "Travel-Chatbot"

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template(
        "Given the following chat history and latest question, create a standalone question that captures the full context:\n\n"
        "Chat History:\n{chat_history}\n\n"
        "Latest Question: {input}\n\n"
        "Standalone question:"
    )
])

ticket_tool = Tool(
    name="get_ticket_info",
    description="Get information about tickets for events and attractions. Input should be the event or attraction name.",
    func=get_ticket_details
)

qa_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template(
        "You are a personalized travel assistant focused on providing tailored travel recommendations. "
        "You have access to tools that can help you get real-time ticket information.\n\n"
        
        "IMPORTANT CONTEXT HANDLING:\n"
        "- The provided context contains personal information about the traveler including:\n"
        "  * Their name and basic information\n"
        "  * Travel preferences and interests\n"
        "  * Dietary restrictions and allergies\n"
        "  * Previous travel experiences\n"
        "  * Special requirements or conditions\n"
        "  * Budget considerations\n"
        "- Always prioritize any health and safety requirements mentioned in the context\n"
        "- Cross-reference preferences with your recommendations\n"
        "- Consider seasonal factors and current travel conditions\n\n"
        
        "RESPONSE GUIDELINES:\n"
        "1. Safety First:\n"
        "   - Account for dietary restrictions and allergies in food recommendations\n"
        "   - Consider accessibility needs in suggested activities\n"
        "   - Include relevant health and safety tips\n\n"
        
        "2. Personalization:\n"
        "   - Reference specific interests from their profile\n"
        "   - Adapt suggestions to their stated preferences\n"
        "   - Consider their past travel experiences\n"
        "   - Stay within mentioned budget constraints\n\n"
        
        "3. Format Your Response:\n"
        "   - Start with personalized greeting using their name\n"
        "   - Provide clear, structured recommendations\n"
        "   - Include specific details (times, prices, locations)\n"
        "   - Always end with a personalized closing \n"
        "   - And make your response to be concise and in a friendly tone\n\n"
        
        "4. Ticket Information:\n"
        "   - When users ask about tickets, use the get_ticket_info tool\n"
        "   - Provide pricing and availability details\n"
        "   - Suggest best booking options\n\n"

        "Context from user profile: {context}\n\n"
        "Previous interactions: {chat_history}\n\n"
        "Current question: {input}\n\n"
        "Personalized recommendation:"
    )
])

def get_rag_chain(model="gemini-1.5-pro"):
    llm = ChatGoogleGenerativeAI(model=model)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Create agent with tools
    agent = create_openai_functions_agent(
        llm=llm,
        prompt=qa_prompt,
        tools=[ticket_tool]
    )
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[ticket_tool],
        verbose=True
    )
    
    # Combine RAG and agent capabilities
    rag_chain = create_retrieval_chain(history_aware_retriever, agent_executor)
    
    return rag_chain