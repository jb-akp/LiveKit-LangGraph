# graph.py
from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv(".env.local")

# -------------------- Build your Interview RAG pipeline --------------------
def create_workflow():
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    pdf_path = os.getenv("COMPANY_PDF_PATH", "./TechCompanyInfo.pdf")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}. Please set COMPANY_PDF_PATH environment variable or place TechCompanyInfo.pdf in the current directory.")

    pages = PyPDFLoader(pdf_path).load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pages_split = text_splitter.split_documents(pages)

    persist_directory = os.getenv("CHROMA_DIR", "./chroma_store")
    os.makedirs(persist_directory, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="company_info",
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    @tool
    def company_info_tool(query: str) -> str:
        """Searches the company information document and returns relevant chunks about the company."""
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found in the company documents."
        # Build the result string step by step for beginners
        result_parts = []
        for i, doc in enumerate(docs):
            info_number = i + 1
            content = doc.page_content
            formatted_info = f"Info {info_number}:\n{content}"
            result_parts.append(formatted_info)
        
        # Join all parts with double newlines
        return "\n\n".join(result_parts)

    @tool
    def record_answer_tool(answer: str) -> str:
        """Records the candidate's answer to a text file for later review."""
        # Simply write the answer to the file
        with open("interview_answers.txt", "a", encoding="utf-8") as f:
            f.write(f"\nAnswer:\n{answer}\n")
            f.write("-" * 50 + "\n")
        
        print(f"Recorded answer: {answer[:50]}...")
        return "Answer recorded successfully!"

    tools = [company_info_tool, record_answer_tool]
    llm = llm.bind_tools(tools)

    class InterviewState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    def decide_next_action(state: InterviewState) -> str:
        """Decide what to do next: tool_executor or end"""
        last = state["messages"][-1]
        
        # Check if we need to execute tool calls from the LLM
        if hasattr(last, "tool_calls") and last.tool_calls and len(last.tool_calls) > 0:
            return "tool_executor"
        
        # Default to end if no specific action needed
        return "end"

    def call_llm(state: InterviewState) -> InterviewState:
        """Main LLM call that handles the interview conversation."""
        system_prompt = (
            "You are a professional interviewer conducting a job interview. "
            "You will ask structured questions in this order:\n"
            "1. First: 'Hello! Thank you for joining us today. Let's start with the basics - could you tell me about yourself? Please share your background, what you're passionate about, and what brings you here today.'\n"
            "2. After they respond: 'That's great to hear! Now, I'd love to learn about your technical background. Could you tell me about your experience with technology? What technologies, programming languages, or technical projects have you worked with?'\n"
            "3. After they respond: 'Excellent! Now, I'd like to hear about a time when you faced a significant challenge, either technical or professional. Could you walk me through the situation, what obstacles you encountered, and how you overcame them? What did you learn from that experience?'\n"
            "4. After they respond: 'Thank you for sharing that with me. Now, I'd like to give you the opportunity to ask me anything about our company, the role, or anything else you'd like to know. What questions do you have for me?'\n\n"
            "IMPORTANT ROUTING RULES:\n"
            "- When the candidate asks questions about the company (mission, culture, revenue, etc.), use the company_info_tool to find relevant information\n"
            "- When the candidate gives answers to your interview questions, use the record_answer_tool to record their response, then acknowledge it and ask the next question\n"
            "- Be conversational, professional, and helpful\n"
            "- If you don't have specific company information, say so honestly and offer to connect them with someone who might know more"
        )
        
        msgs = [SystemMessage(content=system_prompt)] + list(state["messages"])
        message = llm.invoke(msgs)
        return {"messages": [message]}

    def tool_executor(state: InterviewState) -> InterviewState:
        """Execute tool calls from the LLM's response."""
        # Get the tool calls from the last message
        tool_calls = state["messages"][-1].tool_calls
        results = []

        # Go through each tool call one by one
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            
            print(f"Running tool: {tool_name}")

            # Call the right tool based on its name
            if tool_name == "company_info_tool":
                result = company_info_tool.invoke(tool_args)
            elif tool_name == "record_answer_tool":
                result = record_answer_tool.invoke(tool_args)
            else:
                result = f"Unknown tool: {tool_name}"

            # Create a message with the result
            tool_message = ToolMessage(
                tool_call_id=tool_call["id"],
                name=tool_name,
                content=str(result),
            )
            results.append(tool_message)

        print("All tools finished running.")
        return {"messages": results}


    # Build the interview graph
    graph = StateGraph(InterviewState)
    
    # Add nodes
    graph.add_node("llm", call_llm)
    graph.add_node("tool_executor", tool_executor)
    
    # Set up the flow
    graph.set_entry_point("llm")
    
    # Conditional edges from LLM
    graph.add_conditional_edges(
        "llm", 
        decide_next_action, 
        {
            "tool_executor": "tool_executor",
            "end": END
        }
    )
    
    # From tool_executor back to LLM
    graph.add_edge("tool_executor", "llm")

    return graph.compile()
