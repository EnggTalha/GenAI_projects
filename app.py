# ============================================
# Streamlit UI Application - Enhanced Design
# ============================================

import streamlit as st
from main import get_pdf_text, get_text_chunks, get_vector_store, get_answer
import os


# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="Chat with Multiple PDFs",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "### Chat with Multiple PDFs\nPowered by Groq LLM & LangChain"
    }
)


# ============================================
# Enhanced Custom CSS
# ============================================

st.markdown("""
    <style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
    }
    
    /* Welcome box */
    .welcome-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.5);
    }
    .welcome-title {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .welcome-text {
        font-size: 1.1rem;
        line-height: 1.8;
        color: #2d3748;
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15);
    }
    
    /* Chat message styling */
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
    }
    .message-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        margin-left: auto;
        max-width: 70%;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
    }
    .message-assistant {
        background: #f7fafc;
        color: #2d3748;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        margin-right: auto;
        max-width: 70%;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Status boxes */
    .status-box {
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        font-weight: 500;
    }
    .status-success {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
    }
    .status-info {
        background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
        color: white;
    }
    .status-warning {
        background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
        color: white;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        border: none;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Input styling */
    .stTextInput>div>div>input {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem 1rem;
    }
    .stTextInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f7fafc 0%, #edf2f7 100%);
    }
    
    /* File uploader styling */
    .uploadedFile {
        background: white;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #e2e8f0;
    }
    
    /* Progress indicators */
    .progress-step {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem;
        background: white;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Stats cards */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-top: 4px solid #667eea;
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin: 0.5rem 0;
    }
    .stat-label {
        color: #718096;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Scrollbar styling */
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    .chat-container::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 10px;
    }
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================
# Welcome Message
# ============================================

def show_welcome_message():
    """Display enhanced welcome message"""
    st.markdown("""
        <div class="welcome-box">
            <div class="welcome-title">👋 Welcome to Chat with Multiple PDFs!</div>
            <div class="welcome-text">
                <p style="font-size: 1.2rem; margin-bottom: 1.5rem;"><strong>Transform your PDF documents into an interactive Q&A experience</strong></p>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1.5rem 0;">
                    <div class="card">
                        <h3 style="color: #667eea; margin-top: 0;">📤 Upload</h3>
                        <p>Upload one or more PDF documents using the sidebar</p>
                    </div>
                    <div class="card">
                        <h3 style="color: #667eea; margin-top: 0;">⚙️ Process</h3>
                        <p>Click "Process PDFs" to analyze and index your documents</p>
                    </div>
                    <div class="card">
                        <h3 style="color: #667eea; margin-top: 0;">💬 Ask</h3>
                        <p>Type your questions and get accurate answers instantly</p>
                    </div>
                    <div class="card">
                        <h3 style="color: #667eea; margin-top: 0;">🎯 Accurate</h3>
                        <p>Answers are based only on your uploaded PDF content</p>
                    </div>
                </div>
                
                <div style="background: white; padding: 1.5rem; border-radius: 12px; margin-top: 1.5rem;">
                    <h4 style="color: #667eea; margin-top: 0;">✨ Key Features:</h4>
                    <ul style="margin: 0; padding-left: 1.5rem; line-height: 2;">
                        <li><strong>Multi-PDF Support:</strong> Upload and query multiple documents simultaneously</li>
                        <li><strong>Smart Search:</strong> Advanced semantic search finds relevant information quickly</li>
                        <li><strong>Fast Responses:</strong> Powered by Groq's lightning-fast LLM</li>
                        <li><strong>Accurate Answers:</strong> Responses based strictly on your PDF content</li>
                    </ul>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


# ============================================
# Enhanced Sidebar
# ============================================

def show_sidebar():
    """Display enhanced sidebar with PDF upload functionality"""
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; padding: 1rem 0;">
                <h2 style="color: #667eea; margin: 0;">📂 Document Manager</h2>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        
        # File uploader
        pdf_docs = st.file_uploader(
            "**Upload PDF Files**",
            type="pdf",
            accept_multiple_files=True,
            help="Select one or more PDF files to upload"
        )
        
        if pdf_docs:
            st.markdown(f"""
                <div class="status-box status-info">
                    <strong>📄 {len(pdf_docs)} file(s) selected</strong>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Uploaded Files:**")
            for i, pdf in enumerate(pdf_docs, 1):
                file_size = len(pdf.getvalue()) / 1024  # Size in KB
                st.markdown(f"""
                    <div class="uploadedFile">
                        <strong>{i}. {pdf.name}</strong><br>
                        <small style="color: #718096;">{file_size:.1f} KB</small>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Process button
        if st.button("🔄 Process PDFs", type="primary", use_container_width=True):
            if not pdf_docs:
                st.error("⚠️ Please upload at least one PDF file first!")
            else:
                # Progress tracking
                progress_bar = st.progress(0)
                status_container = st.empty()
                
                try:
                    # Step 1: Extract text
                    progress_bar.progress(10)
                    status_container.markdown("""
                        <div class="progress-step">
                            <span style="font-size: 1.5rem;">📄</span>
                            <div>
                                <strong>Step 1/3: Extracting text...</strong><br>
                                <small>Reading PDF content</small>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    raw_text = get_pdf_text(pdf_docs)
                    
                    if not raw_text.strip():
                        status_container.error("❌ No text could be extracted from the PDFs.")
                    else:
                        text_length = len(raw_text)
                        progress_bar.progress(40)
                        status_container.markdown(f"""
                            <div class="status-box status-success">
                                ✅ Extracted {text_length:,} characters from {len(pdf_docs)} PDF(s)
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Step 2: Split chunks
                        progress_bar.progress(50)
                        status_container.markdown("""
                            <div class="progress-step">
                                <span style="font-size: 1.5rem;">✂️</span>
                                <div>
                                    <strong>Step 2/3: Creating chunks...</strong><br>
                                    <small>Splitting text for optimal search</small>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        text_chunks = get_text_chunks(raw_text)
                        progress_bar.progress(70)
                        status_container.markdown(f"""
                            <div class="status-box status-success">
                                ✅ Created {len(text_chunks)} text chunks
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Step 3: Vector store
                        progress_bar.progress(80)
                        status_container.markdown("""
                            <div class="progress-step">
                                <span style="font-size: 1.5rem;">🔍</span>
                                <div>
                                    <strong>Step 3/3: Creating embeddings...</strong><br>
                                    <small>Building searchable index</small>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        get_vector_store(text_chunks)
                        progress_bar.progress(100)
                        
                        # Success
                        st.session_state['pdfs_processed'] = True
                        st.session_state['num_chunks'] = len(text_chunks)
                        st.session_state['num_pdfs'] = len(pdf_docs)
                        st.session_state['text_length'] = text_length
                        
                        status_container.markdown(f"""
                            <div class="status-box status-success">
                                <h3 style="margin: 0 0 0.5rem 0;">✅ Processing Complete!</h3>
                                <p style="margin: 0;">Your documents are ready for questions</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Show stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("PDFs", len(pdf_docs))
                        with col2:
                            st.metric("Chunks", len(text_chunks))
                        with col3:
                            st.metric("Characters", f"{text_length:,}")
                        
                        progress_bar.empty()
                        
                except Exception as e:
                    status_container.error(f"❌ Error: {str(e)}")
                    progress_bar.empty()
        
        st.markdown("---")
        
        # Status indicator
        if st.session_state.get('pdfs_processed', False):
            st.markdown("""
                <div class="status-box status-success">
                    <strong>✅ Ready to Chat!</strong><br>
                    <small>Your PDFs are processed and indexed</small>
                </div>
            """, unsafe_allow_html=True)
            
            # Show stats
            if 'num_pdfs' in st.session_state:
                st.markdown("**📊 Processing Stats:**")
                st.json({
                    "PDFs Processed": st.session_state.get('num_pdfs', 0),
                    "Text Chunks": st.session_state.get('num_chunks', 0),
                    "Total Characters": f"{st.session_state.get('text_length', 0):,}"
                })
        else:
            st.markdown("""
                <div class="status-box status-warning">
                    <strong>ℹ️ Get Started</strong><br>
                    <small>Upload PDFs and click "Process PDFs"</small>
                </div>
            """, unsafe_allow_html=True)


# ============================================
# Enhanced Chat Interface
# ============================================

def show_chat_interface():
    """Display enhanced chat interface"""
    # Header
    st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
            <h2 style="color: #667eea; margin: 0;">💬 Chat with Your Documents</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Check if PDFs are processed
    if not st.session_state.get('pdfs_processed', False):
        st.markdown("""
            <div class="card" style="text-align: center; padding: 3rem;">
                <span style="font-size: 4rem;">📚</span>
                <h3 style="color: #667eea;">No Documents Processed Yet</h3>
                <p style="color: #718096;">Please upload and process PDFs in the sidebar to start asking questions.</p>
            </div>
        """, unsafe_allow_html=True)
        return
    
    # Check if vector store exists
    if not os.path.exists("faiss_index"):
        st.warning("⚠️ Vector store not found. Please process PDFs again.")
        st.session_state['pdfs_processed'] = False
        return
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Chat history header
        if st.session_state.chat_history:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown("### 📜 Conversation History")
            with col2:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
            
            st.markdown("---")
            
            # Display messages
            for i, (question, answer) in enumerate(reversed(st.session_state.chat_history)):
                # User message
                st.markdown(f"""
                    <div class="message-user">
                        <strong>You:</strong><br>
                        {question}
                    </div>
                """, unsafe_allow_html=True)
                
                # Assistant message
                st.markdown(f"""
                    <div class="message-assistant">
                        <strong>🤖 Assistant:</strong><br>
                        {answer}
                    </div>
                """, unsafe_allow_html=True)
                
                if i < len(st.session_state.chat_history) - 1:
                    st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("---")
    
    # Question input area
    st.markdown("### Ask a Question")
    
    # Input with better styling
    user_question = st.text_input(
        "",
        placeholder="Type your question here... (e.g., What is the main topic discussed in the PDF?)",
        key="question_input",
        label_visibility="collapsed"
    )
    
    # Submit button
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        submit_button = st.button("🔍 Ask", type="primary", use_container_width=True)
    with col2:
        if st.button("🔄 New Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    # Process question
    if submit_button and user_question:
        if not user_question.strip():
            st.warning("⚠️ Please enter a valid question.")
        else:
            with st.spinner("🔍 Searching through your documents and generating answer..."):
                try:
                    answer = get_answer(user_question)
                    
                    # Add to chat history
                    st.session_state.chat_history.append((user_question, answer))
                    
                    # Rerun to show new message
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")
                    st.info("💡 Try processing the PDFs again or check your Groq API key.")


# ============================================
# Main Application
# ============================================

def main():
    """Main application function"""
    # Enhanced header
    st.markdown("""
        <div class="main-header">
            <h1>📚 Chat with Multiple PDFs</h1>
            <p>Ask questions and get answers from your PDF documents using AI</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Welcome message (collapsible)
    with st.expander("👋 Welcome & Instructions", expanded=False):
        show_welcome_message()
    
    # Main layout
    col1, col2 = st.columns([2.5, 1])
    
    with col1:
        show_chat_interface()
    
    # Sidebar is automatically shown
    show_sidebar()


# ============================================
# Run Application
# ============================================

if __name__ == "__main__":
    main()
