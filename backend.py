from __future__ import annotations

import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph , START,END
from typing import TypedDict , Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver # we will use sqlite saver to store the conversation history in a sqlite database,and it is not built in langgraph so we have to import it from langgraph.checkpoint.sqlite(we are externally downloading it from langgraph).
                                                    # we are using sqlite saver because it is a simple and lightweight database that can be easily integrated with our chatbot application and it will help us to store the conversation history of each thread in a structured way and we can easily retrieve it when needed.after refreshing the page the conversation history will not be lost because it is stored in the database and we can retrieve it using the thread id.        

from langgraph.graph.message import add_messages # reducer function
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import requests
import sqlite3   # we need to make a database connection to create a database and a table to store the conversation history of each thread and we will use sqlite3 library to do that.
import os
import uuid



load_dotenv()

#.............................................................................................................................................
# LLM
api_key = os.getenv("GROQ_API_KEY")

llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0
)



from functools import lru_cache

@lru_cache
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embeddings()

#.............................................................................................................................................
# 2. PDF retriever store (per thread)


_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


#.............................................................................................................................................
# Tools



@tool
def search_tool(query: str) -> str:
    """
    Use this tool ONLY when:
    - User asks about current events
    - Needs factual information from the internet

    DO NOT use for greetings, casual conversation, math, or stock prices.
    """
    return DuckDuckGoSearchRun(region="us-en").run(query)


@tool
def calculator(first_num: float, second_num: float, operation: str) -> str:
    """
    Use ONLY for mathematical calculations.
    Supported operations: add, sub, mul, div.
    Do NOT use for general questions.
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return f"Result: {result}"
        
        
    except Exception as e:
        return {"error": str(e)}
    



@tool
def get_stock_price(company: str) -> str:
    """
    Get stock price of a company using name.
    """
    try:
        # Step 1: find symbol using search
        search_query = f"{company} stock ticker symbol"
        search_result = DuckDuckGoSearchRun().run(search_query)

        # simple extraction (hacky but works)
        words = search_result.split()
        symbol = None

        for word in words:
            if word.isupper() and len(word) <= 5:
                symbol = word
                break

        if not symbol:
            symbol = company.upper()

        # Step 2: call API
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=EC2VBTMOVA408PKR"
        r = requests.get(url)
        data = r.json()

        quote = data.get("Global Quote", {})

        if not quote:
            return f"Could not find stock data for {company}"

        price = quote.get("05. price", "N/A")

        return f"{company} ({symbol}) stock price is {price} USD"

    except Exception as e:
        return f"Error: {str(e)}"
    

@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)
#.............................................................................................................................................
# state

class chatstate(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
#.............................................................................................................................................
# Nodes



def chat_node(state: chatstate, config=None):
    messages = state['messages']
    thread_id = config.get("configurable", {}).get("thread_id", "") if config else ""

    system_prompt = SystemMessage(content=f"""
You are a helpful AI assistant with access to tools.
Current thread_id: {thread_id}

STRICT RULES:
- Stock price → call get_stock_price ONLY ONCE
- Math → call calculator ONLY ONCE  
- PDF questions → call rag_tool with thread_id="{thread_id}" ONLY ONCE
- If tool result is already in messages → give final answer, DO NOT call tool again
- Greetings → normal response, no tools
""")

    last_message = messages[-1]

    # After a tool result → answer using tool output (still include system prompt!)
    if last_message.__class__.__name__ == "ToolMessage":
        response = llm_with_tools.invoke([system_prompt] + messages)
    else:
        response = llm_with_tools.invoke([system_prompt] + messages)

    return {'messages': [response]}

tool_node = ToolNode(tools)
#.............................................................................................................................................
#Checkpoint

conn=sqlite3.connect(database='chat_history.db', check_same_thread=False) # we will create a database named chat_history.db to store the conversation history of each thread in a table named conversations and we will use the thread id as the primary key to store the conversation history of each thread in a structured way and we can easily retrieve it when needed.
                                                            # we are using check_same_thread=False because we are using sqlite in a multi-threaded environment and it will allow us to use the same database connection across multiple threads without any issues.if we put true then it will not allow us to use the same database connection across multiple threads and we will get an error when we try to access the database from multiple threads.

cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS thread_titles (
    thread_id TEXT PRIMARY KEY,
    title TEXT
)
""")
conn.commit()


checkpointer = SqliteSaver(conn=conn)

#..............................................................................................................................................
#graph structure

graph = StateGraph(chatstate)
graph.add_node('chat_node',chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START,'chat_node')
# if the LLM asked for a tool, go to ToolNode; else finish
graph.add_conditional_edges("chat_node", tools_condition)

graph.add_edge("tools", "chat_node")

@lru_cache
def get_chatbot():
    return graph.compile(checkpointer=checkpointer)

Chatbot = get_chatbot()

#.............................................................................................................................................




def retrieve_all_threads():
    cursor = conn.cursor()
    
    cursor.execute("SELECT thread_id, title FROM thread_titles")
    
    rows = cursor.fetchall()
    
    return {thread_id: title for thread_id, title in rows}





def save_thread_title(thread_id, title):
    cursor = conn.cursor()
    
    cursor.execute("""
    INSERT OR REPLACE INTO thread_titles (thread_id, title)
    VALUES (?, ?)
    """, (thread_id, title))
    
    conn.commit()

def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})