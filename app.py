import streamlit as st
import os
import time
from dotenv import load_dotenv
from src.document_loader import load_document
from src.vector_store import create_vector_store
from src.rag_chain import get_rag_chain
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Agentic Research Assistant",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Premium UI / CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0F1116 0%, #090B10 100%);
        color: #E6EDF3;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #21262D;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #FFFFFF;
        font-weight: 700;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0D1117;
    }
    ::-webkit-scrollbar-thumb {
        background: #30363D;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #58A6FF;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(135deg, #238636 0%, #2EA043 100%);
        color: white;
        border: 1px solid rgba(255,255,255,0.1);
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(46, 160, 67, 0.4);
        border-color: rgba(255,255,255,0.2);
    }
    .stButton > button:active {
        transform: translateY(0px);
    }

    /* Secondary Button (Clear Chat) */
    div[data-testid="stVerticalBlock"] > div > div > div > div > .stButton > button {
        /* Default styling covers this mostly, but we can target specific buttons if we add classes or IDs, 
           for now we keep the uniform premium look */
    }

    /* Inputs */
    .stTextInput > div > div > input {
        background-color: #0D1117;
        border: 1px solid #30363D;
        color: #E6EDF3;
        border-radius: 6px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #58A6FF;
        box-shadow: 0 0 0 1px #58A6FF;
    }

    /* Chat Input */
    .stChatInput > div > div > div > input {
        background-color: #0D1117;
        color: white;
    }

    /* Header */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: -webkit-linear-gradient(120deg, #FFFFFF, #8B949E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* Document Success/Status */
    .stStatusWidget {
        background-color: #161B22;
        border: 1px solid #30363D;
    }

    /* Chat Messages */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatMessage.user {
        background-color: rgba(56, 139, 253, 0.1);
        border: 1px solid rgba(56, 139, 253, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Caching & Optimization
# -----------------------------------------------------------------------------
@st.cache_resource
def get_embedding_model():
    """
    Loads and caches the embedding model to improve speed.
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    # API Key Management
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.markdown("### 🔑 Credentials")
        api_key = st.text_input("Groq API Key", type="password", help="Enter your Groq API Key to proceed.")

    st.markdown("---")
    
    # Document Management
    st.markdown("### 📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a file (PDF, TXT, CSV, DOCX)", 
        type=['pdf', 'csv', 'docx', 'xlsx', 'txt'],
        help="Upload the document you want to chat with."
    )
    
    process_col1, process_col2 = st.columns(2)
    with process_col1:
        process_button = st.button("⚡ Process Details") # Renamed for flair
    with process_col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    st.caption("Agentic Research Assistant v2.0")

# -----------------------------------------------------------------------------
# Main Application Logic
# -----------------------------------------------------------------------------
st.title("✨ Agentic Research Assistant")
st.markdown("Chat with your documents at lightning speed.")

# Session state initialization
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Process document
if process_button and uploaded_file and api_key:
    with st.status("Processing document...", expanded=True) as status:
        try:
            start_time = time.time()
            
            st.write("📥 Loading file...")
            documents = load_document(uploaded_file)
            
            st.write("🧠 Loading embedding model...")
            embeddings = get_embedding_model()
            
            st.write("⚡ Creating vector store...")
            st.session_state.vector_store = create_vector_store(documents, embeddings)
            
            elapsed_time = time.time() - start_time
            status.update(label=f"Done! ({elapsed_time:.2f}s)", state="complete", expanded=False)
            st.rerun()
            
        except Exception as e:
            status.update(label="❌ Error processing document", state="error")
            st.error(f"Error details: {str(e)}")

elif process_button and not api_key:
    st.error("⚠️ Please provide a Groq API Key in the sidebar.")

# -----------------------------------------------------------------------------
# Chat Interface
# -----------------------------------------------------------------------------
if st.session_state.vector_store:
    # Initialize RAG chain with the API key
    rag_chain = get_rag_chain(st.session_state.vector_store, api_key)

    # Display Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Quick Actions
    st.caption("Quick Actions")
    q_col1, q_col2, q_col3 = st.columns(3)
    quick_prompt = None
    
    if q_col1.button("📝 Summarize", help="Generate a concise summary"):
        quick_prompt = "Summarize the document in 5 concise bullet points."
    
    if q_col2.button("🔑 Key Insights", help="Extract main points"):
        quick_prompt = "What are the top 3 key insights from this document?"
        
    if q_col3.button("❓ Suggest Questions", help="Get ideas for what to ask"):
        quick_prompt = "Suggest 3 interesting questions I can ask about this document."

    # User Input
    # Check if a quick button was clicked OR user typed input
    user_input = st.chat_input("Ask something about your document...")
    
    if quick_prompt:
        prompt = quick_prompt
    else:
        prompt = user_input

    if prompt:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Streamed Response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                # Assuming invoke returns the full dict, wait - we want streaming.
                # If the chain supports it, we iterate. 
                # Note: The current rag_chain returns a specific dict structure.
                # Let's try to stream. If the underlying chain is runnable, it should work.
                
                # We need to pass the same input format
                stream = rag_chain.stream({"input": prompt})
                
                for chunk in stream:
                    # Check what the chunk contains.
                    # Usually 'answer' key has the delta string
                    if "answer" in chunk:
                        content = chunk["answer"]
                        full_response += content
                        response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                
                # Append to history
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                # Fallback to invoke if streaming fails or structure is different
                try:
                    response = rag_chain.invoke({"input": prompt})
                    answer = response["answer"]
                    response_placeholder.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    full_response = answer
                except Exception as inner_e:
                     st.error(f"Error generating response: {inner_e}")

            # Source Documents (Optional - in expander)
            # Invoke again or use the context from the last chunk if available?
            # Streaming context is tricky. Often context comes in the first chunk or all chunks.
            # To be safe and simple: We might skip showing sources in streaming mode OR
            # we just do a separate retrieval call if we really want them, BUT
            # let's try to grab context from the final response object if possible.
            # Actually, standard LangChain streaming yields chunks. 
            # If we want context, we might have to rely on the 'invoke' fallback OR
            # check if one of the chunks has 'context'.
            
            # For now, to keep it fast, we'll only show sources if we have them.
            # The previous code showed sources. Let's try to preserve that feature if possible.
            # But making it FASTER is the priority. Computing sources + answer is fine.
            
            # Re-invoking just for sources is wasteful. 
            # Let's stick to the answer for now. 
            # (If the user *really* needs sources, we can add a toggle "Show Sources" which uses invoke)
            
else:
    # Welcome / Call to Action
    if not uploaded_file:
        st.info("👈 Upload a document in the sidebar to get started!")
    elif not st.session_state.vector_store:
        st.info("👈 Click 'Process Details' to analyze your document.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <small>Powered by Groq & LangChain | Optimized for Speed</small>
    </div>
    """, 
    unsafe_allow_html=True
)
