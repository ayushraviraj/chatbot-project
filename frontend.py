import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

from backend import (
    Chatbot,
    ingest_pdf,
    retrieve_all_threads,
    save_thread_title,
    thread_document_metadata,
)

#********************************************************************************************************************************************************************************************************************

def grenerate_thread_id():  # we will generate a unique thread id for each conversation using uuid library and return it
    thread_id = uuid.uuid4()
    return str(thread_id)


def reset_chat():  # when we click on new chat button then it will reset the chat and generate a new thread id and add that thread id to the chat threads list and set the message history to empty list
    st.session_state['message_history'] = []
    thread_id = grenerate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])




def add_thread(thread_id): # we will add the thread id to the chat threads list if it's not already present and we will use this function when we click on new chat button to add the new thread id to the chat threads list and also when we load the conversation history of a thread then we will add that thread id to the chat threads list if it's not already present so that it will be displayed in the sidebar of the chatbot application
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'][thread_id] = "New Chat"


def load_conversation(thread_id):  # we will get the conversation history of the thread from the chatbot state and return it and we will use this function when we click on the thread id in the sidebar to load the conversation history of that thread and display it in the chat interface
    return Chatbot.get_state(config={"configurable": {"thread_id": thread_id}}).values.get("messages", []) # we will get the conversation history of the thread from the chatbot state and return it    

#********************************************************************************************************************************************************************************************************************

#this will store the conversation history in dictionary format


if 'message_history' not in st.session_state:   # we will not using this because when we press enter then in streamlit reruns the whole code and the message history will be lost so we will use session state to store the message history        
    st.session_state['message_history'] = []    # session state is a dictionary that can store any data and it will persist across reruns of the app,in this key is message_history and value is an empty list


if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = grenerate_thread_id()  # we will generate a unique thread id for each conversation and store it in session state

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()   # we will store the conversation history of each thread in a dictionary where key is thread id and value is message history of that thread

if 'ingested_docs' not in st.session_state:  # this will store the ingested document metadata for each thread in a dictionary where key is thread id and value is another dictionary where key is document name and value is document metadata
    st.session_state['ingested_docs'] = {}

add_thread(st.session_state['thread_id']) # we will add the thread id to the chat threads list if it's not already present  

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = list(st.session_state["chat_threads"].items())[::-1]
selected_thread = None


#********************************************************************************************************************************************************************************************************************

st.sidebar.title("Chatbot")


st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

st.sidebar.subheader("Past conversations")
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
   for thread_id, title in threads:
    if st.sidebar.button(str(title), key=f"side-thread-{thread_id}"):
        selected_thread = thread_id

#********************************************************************************************************************************************************************************************************************
st.title("Multi Utility Chatbot")

# Chat area
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Ask about your document or use tools")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, _ in Chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    if not getattr(message_chunk, "tool_calls", None):
                        yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

st.divider()

if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        if msg.content:
            temp_messages.append({"role": role, "content": msg.content})

    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()