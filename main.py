# ============================================
# Core Logic Module - PDF Processing & QA
# ============================================

from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv

# LangChain utilities
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Groq LLM
from langchain_groq import ChatGroq

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================
# Load Environment Variables
# ============================================

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================
# Extract Text from Uploaded PDFs
# ============================================

def get_pdf_text(pdf_docs):
    """
    Reads multiple PDF files and extracts text from each page.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# ============================================
# Split Large Text into Chunks
# ============================================

def get_text_chunks(text):
    """
    Splits text into overlapping chunks
    to preserve context for embeddings.
    Optimized for better semantic search.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Slightly smaller chunks for better precision
        chunk_overlap=150,  # Good overlap to maintain context
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Better splitting strategy
    )
    return splitter.split_text(text)


# ============================================
# Create & Save FAISS Vector Store
# ============================================

def get_vector_store(text_chunks):
    """
    Converts text chunks into embeddings
    and stores them locally using FAISS.
    Uses normalized embeddings for better semantic search.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}  # Normalize embeddings for better cosine similarity
    )

    vector_store = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    vector_store.save_local("faiss_index")


# ============================================
# Build Question-Answering Chain
# ============================================

def get_llm():
    """
    Creates and returns Groq LLM instance.
    """
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",  # Using 8b-instant as it's currently supported
        temperature=0.1  # Lower temperature for more accurate responses
    )
    return llm


# ============================================
# Handle User Question
# ============================================

def get_answer(user_question):
    """
    Loads vector store, finds relevant chunks using semantic search,
    and generates an answer using Groq LLM.
    Returns the answer text.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}  # Normalize for better cosine similarity
    )

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Use MMR (Maximal Marginal Relevance) for better semantic diversity
    # This balances relevance with diversity to avoid redundant information
    try:
        # Try MMR first for better diversity
        docs = db.max_marginal_relevance_search(
            user_question, 
            k=6,  # Number of documents to retrieve
            fetch_k=20,  # Fetch more candidates for MMR
            lambda_mult=0.5  # Balance between relevance (0) and diversity (1)
        )
    except:
        # Fallback to similarity search with scores if MMR fails
        docs_with_scores = db.similarity_search_with_score(user_question, k=10)
        
        # Filter by relevance score (lower = more similar for cosine distance)
        # For normalized embeddings, good matches typically have scores < 0.7
        filtered_docs = []
        for doc, score in docs_with_scores:
            if score < 0.7:  # More strict threshold for better quality
                filtered_docs.append(doc)
        
        # If no good matches, use top 6 regardless of score
        if not filtered_docs:
            docs = [doc for doc, score in docs_with_scores[:6]]
        else:
            docs = filtered_docs[:6]
    
    # Format context from documents with chunk numbering
    context = "\n\n".join([f"[Document Chunk {i+1}]:\n{doc.page_content}" for i, doc in enumerate(docs)])
    
    # Create prompt template
    prompt_template = """
    SYSTEM PROMPT:
    You are a precise document analysis assistant. Your ONLY source of information is the provided PDF context.
    
    CRITICAL RULES:
    1. Answer ONLY using information explicitly stated in the provided context.
    2. Be ACCURATE - quote or paraphrase directly from the context.
    3. Be CONCISE - provide direct answers without unnecessary elaboration.
    4. If the answer is NOT in the context, respond EXACTLY: "The answer is not available in the uploaded PDF documents."
    5. Do NOT use external knowledge, assumptions, or general information.
    6. Do NOT make inferences beyond what is explicitly stated.
    7. If the context is insufficient, state that clearly.
    
    Context from PDF:
    {context}

    User Question:
    {question}

    Answer (based ONLY on PDF context):
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Get LLM and invoke
    llm = get_llm()
    formatted_prompt = prompt.format(context=context, question=user_question)
    response = llm.invoke(formatted_prompt)
    
    return response.content