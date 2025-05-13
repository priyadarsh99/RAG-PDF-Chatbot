import streamlit as st
import os
import tempfile
import shutil
from dotenv import load_dotenv
from chromadb.config import Settings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = hf_token  # Required for HuggingFaceEmbeddings

# Embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Chroma settings for local usage
chroma_settings = Settings(chroma_api_impl="local", anonymized_telemetry=False)

# Streamlit UI
st.set_page_config(page_title="Conversational PDF RAG", layout="wide")
st.title("📄 Conversational RAG for PDFs with Chat History")

# Sidebar Inputs
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Your GROQ API KEY", type="password")
    session_id = st.text_input("Session ID", value="default_session")
    uploaded_files = st.file_uploader("Choose PDF file(s)", type="pdf", accept_multiple_files=True)

    if st.button("Clear Chroma DB"):
        if os.path.exists('./chroma_db'):
            shutil.rmtree('./chroma_db')
            st.success("✅ Chroma vector store cleared.")
        else:
            st.info("No Chroma DB found to clear.")

if not api_key:
    st.warning("Please enter the GROQ API key.")
    st.stop()

llm = ChatGroq(groq_api_key=api_key, model="llama-3.3-70b-versatile")

# Store chat history in Streamlit state
if "store" not in st.session_state:
    st.session_state.store = {}

if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)

    # Create Chroma vector store
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    retriever = vector_store.as_retriever()

    # Contextual question rephrasing
    contextualize_q_system_prompt = (
        "You are given a chat history and a follow-up user question that may depend on previous conversation context. "
        "Your task is to rewrite the user's latest question into a standalone version that can be understood without referring to the prior chat. "
        "If the question is already standalone, return it unchanged. Do not answer the question—only reformulate it if necessary."
    )


    contextualise_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualise_q_prompt
    )

    # QA chain
    system_prompt = (
    "You are a knowledgeable and helpful AI assistant. "
    "Using only the information provided in the retrieved context below, answer the user's question clearly and accurately. "
    "If the answer is not explicitly stated or cannot be inferred with high confidence from the context, respond with \"I don't know.\" "
    "Keep your answers concise—no more than three sentences.\n\nContext:\n{context}"
    )


    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # History manager
    def get_session_history(_: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # Chat Interface
    st.subheader("Ask questions about the uploaded PDF(s)")
    # Display previous messages
    if session_id in st.session_state.store:
        with st.container():
            st.markdown("### 💬 Chat History")
            for msg in st.session_state.store[session_id].messages:
                role = msg.type  # 'human' or 'ai'
                with st.chat_message("user" if role == "human" else "assistant"):
                    st.markdown(msg.content)

    user_input = st.chat_input("Please type your query:")

    if user_input:
        session_history = get_session_history(session_id)

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            try:
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}},
                )
                answer = response["answer"]
                with st.chat_message("assistant"):
                    st.markdown(answer)

                # Show retrieved documents if any
                if "context" in response:
                    with st.expander("🔎 Retrieved Context"):
                        for i, doc in enumerate(response["context"]):
                            source = doc.metadata.get("source", f"Chunk {i+1}")
                            st.markdown(f"**Source:** {source}")
                            st.markdown(doc.page_content[:500] + "...")
            except Exception as e:
                st.error(f"Error during response generation: {e}")

else:
    st.info("📂 Please upload at least one PDF to begin.")
