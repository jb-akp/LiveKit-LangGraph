# rag_graph.py
from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()

# -------------------- Build your RAG pipeline --------------------
def create_workflow():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    pdf_path = "/Users/jimmybradford/Downloads/Stock_Market_Performance_2024.pdf"
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pages = PyPDFLoader(pdf_path).load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pages_split = text_splitter.split_documents(pages)

    persist_directory = os.getenv("CHROMA_DIR", "./chroma_store")
    os.makedirs(persist_directory, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="stock_market",
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    @tool
    def retriever_tool(query: str) -> str:
        """Searches the Stock Market Performance 2024 document and returns relevant chunks."""
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant passages found in the document."
        return "\n\n".join([f"Doc {i+1}:\n{d.page_content}" for i, d in enumerate(docs)])

    tools = [retriever_tool]
    llm = llm.bind_tools(tools)

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        return hasattr(last, "tool_calls") and len(last.tool_calls) > 0

    system_prompt = (
        "You are an assistant that answers questions about Stock Market Performance in 2024 "
        "using the loaded PDF. Use the `retriever_tool` as needed, and cite passages you used."
    )

    tools_dict = {t.name: t for t in tools}

    def call_llm(state: AgentState) -> AgentState:
        msgs = [SystemMessage(content=system_prompt)] + list(state["messages"])
        message = llm.invoke(msgs)
        return {"messages": [message]}

    def take_action(state: AgentState) -> AgentState:
        """Execute tool calls from the LLM's response."""
        tool_calls = state["messages"][-1].tool_calls
        results = []

        for t in tool_calls:
            name = t["name"]
            args = t.get("args", {}) or {}
            print(f"Calling Tool: {name} with args: {args!r}")

            if name not in tools_dict:
                result = f"Unknown tool: {name}"
            else:
                # IMPORTANT: BaseTool.invoke takes a single 'input' argument.
                # For @tool with signature retriever_tool(query: str), pass {"query": "..."} as the input.
                try:
                    result = tools_dict[name].invoke(args)
                except TypeError as e:
                    # Fallbacks for odd payloads (some providers may send a bare string)
                    if isinstance(args, str):
                        result = tools_dict[name].invoke({"query": args})
                    else:
                        raise

            results.append(
                ToolMessage(
                    tool_call_id=t["id"],
                    name=name,
                    content=str(result),
                )
            )

        print("Tools Execution Complete. Back to the model!")
        return {"messages": results}


    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("retriever_agent", take_action)
    graph.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
    graph.add_edge("retriever_agent", "llm")
    graph.set_entry_point("llm")

    # Return the compiled graph (LangGraph Runnable)
    return graph.compile()
