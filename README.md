# 🤖 Multi-Utility AI Chatbot (LangGraph + Streamlit)

An advanced AI chatbot built using **LangGraph, LangChain, and Streamlit** with support for:
- Tool usage (search, calculator, stock price)
- RAG (PDF-based question answering)
- Streaming responses
- Persistent chat history (SQLite)
- Multi-thread conversations

---

## 🚀 Features

### 💬 Conversational AI
- Built using LLM (Groq API with LLaMA 3.3)
- Context-aware responses using LangGraph state

### 🧠 RAG (Retrieval-Augmented Generation)
- Upload PDF and ask questions
- Uses FAISS vector store + HuggingFace embeddings
- Chunking using RecursiveCharacterTextSplitter

### 🔧 Tool Calling System
Integrated tools:
- 🌐 Web Search (DuckDuckGo)
- 🧮 Calculator (custom tool)
- 📈 Stock Price Fetcher (API-based)
- 📄 PDF Query Tool (RAG)

### 🔄 Streaming Responses
- Real-time token streaming in UI
- Smooth user experience using Streamlit

### 🧵 Multi-Thread Chat System
- Each chat has unique thread_id
- Resume previous conversations anytime

### 💾 Persistent Memory
- Chat history stored using SQLite
- Implemented using LangGraph Checkpointer

### 📊 Document Tracking
- Tracks uploaded PDFs per thread
- Shows metadata (chunks, pages, filename)

---

## 🛠 Tech Stack

- Python
- LangGraph
- LangChain
- Streamlit
- FAISS (Vector DB)
- HuggingFace Embeddings
- SQLite (Database)
- Groq API (LLM)
- DuckDuckGo Search API

---

## ⚙️ Architecture Overview

- **LangGraph** → Controls flow (chat → tools → response)
- **ToolNode** → Handles tool execution
- **StateGraph** → Maintains conversation state
- **SQLite Checkpointer** → Stores chat history
- **FAISS** → Stores document embeddings for RAG

---

## ▶️ How to Run Locally

```bash
git clone https://github.com/ayushraviraj/chatbot-project.git
cd chatbot-project

pip install -r requirements.txt
streamlit run frontend.py
