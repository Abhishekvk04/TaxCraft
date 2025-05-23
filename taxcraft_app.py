import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import tempfile
import json

# Set page configuration
st.set_page_config(
    page_title="TaxCraft AI - Indian Tax Assistant",
    page_icon="💼",
    layout="wide"
)

# Application title and description
st.title("TaxCraft AI - Your Personal Tax Assistant")
st.markdown("""
This AI assistant helps Indian citizens with tax planning, deductions, 
exemptions, and other tax-related queries. Upload your tax documents for personalized advice.
""")

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
    
if 'chain' not in st.session_state:
    st.session_state.chain = None

# Sidebar for API key and document uploads
st.sidebar.header("Configuration")

# Google API Key input
api_key = st.sidebar.text_input("Enter Google Generative AI API Key", type="password")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# Document upload section
st.sidebar.header("Upload Documents")
knowledge_base = st.sidebar.file_uploader("Upload Knowledge Base PDF", type="pdf")
form_16a = st.sidebar.file_uploader("Upload Form 16A or Other Tax Documents", type="pdf")

# Function to process uploaded documents
def process_documents():
    if not api_key:
        st.sidebar.error("Please enter your Google API Key")
        return False
    
    if not (knowledge_base or form_16a):
        st.sidebar.warning("Please upload at least one document")
        return False
    
    with st.spinner("Processing documents..."):
        try:
            documents = []
            
            # Process Knowledge Base if uploaded
            if knowledge_base:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_kb:
                    temp_kb.write(knowledge_base.read())
                    kb_path = temp_kb.name
                
                kb_loader = UnstructuredPDFLoader(kb_path)
                kb_content = kb_loader.load()
                documents.extend(kb_content)
                os.unlink(kb_path)  # Delete temp file
            
            # Process Form 16A if uploaded
            if form_16a:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_form:
                    temp_form.write(form_16a.read())
                    form_path = temp_form.name
                
                form_loader = UnstructuredPDFLoader(form_path)
                form_content = form_loader.load()
                documents.extend(form_content)
                os.unlink(form_path)  # Delete temp file
            
            # Split documents into chunks - reduced size to minimize token usage
            splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            chunks = splitter.split_documents(documents)
            
            # Limit number of chunks to reduce API calls
            if len(chunks) > 50:
                chunks = chunks[:50]
                st.warning(f"Document too large. Using first 50 chunks only to stay within API limits.")
            
            # Create embeddings and vector store
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore = Chroma.from_documents(chunks, embeddings)
            
            # Create retrievers with fewer results to reduce API calls
            retriever_vectordb = vectorstore.as_retriever(search_kwargs={"k": 1})
            keyword_retriever = BM25Retriever.from_documents(chunks)
            keyword_retriever.k = 1
            
            # Create ensemble retriever with fewer results
            ensemble_retriever = EnsembleRetriever(
                retrievers=[retriever_vectordb, keyword_retriever],
                weights=[0.5, 0.5]
            )
            
            # Create LLM with proper model name and reduced temperature
            try:
                # List available models
                available_models = []
                if api_key:
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    available_models = [model.name for model in genai.list_models()]
                    st.sidebar.info(f"Available models: {', '.join(available_models)}")
                
                # Choose appropriate model name - try Flash model first (cheaper)
                model_name = "gemini-1.5-flash"  # Faster, cheaper model
                if "gemini-1.5-flash" not in available_models:
                    model_name = "gemini-1.0-pro"  # Fallback to older, cheaper model
                if model_name not in available_models and "gemini-pro" in available_models:
                    model_name = "gemini-pro"
                    
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0.3,  # Lower temperature to reduce token usage
                    convert_system_message_to_human=True,
                    max_output_tokens=512  # Limit response length
                )
                st.sidebar.success(f"Using model: {model_name}")
            except Exception as e:
                st.sidebar.error(f"Error initializing model: {str(e)}")
                raise e
            
            # Create prompt template - simplified to reduce token usage
            template = """
            You are TaxCraft, an AI assistant for Indian tax planning. Use the provided context to answer tax-related questions clearly and concisely.

            Context: {context}
            
            Question: {query}
            
            Answer (keep it concise and relevant to Indian taxes):
            """
            
            prompt = ChatPromptTemplate.from_template(template)
            output_parser = StrOutputParser()
            
            # Create chain
            chain = (
                {"context": ensemble_retriever, "query": RunnablePassthrough()}
                | prompt
                | llm
                | output_parser
            )
            
            # Save components to session state for later use
            st.session_state.chain = chain
            st.session_state.documents_loaded = True
            st.session_state.ensemble_retriever = ensemble_retriever
            st.session_state.llm = llm
            
            return True
            
        except Exception as e:
            st.sidebar.error(f"Error processing documents: {str(e)}")
            return False

# Button to process documents
if st.sidebar.button("Process Documents"):
    success = process_documents()
    if success:
        st.sidebar.success("Documents processed successfully! You can now start chatting.")

# Main chat interface
st.header("Chat with TaxCraft")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_query := st.chat_input("Ask about Indian taxes..."):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        if not st.session_state.documents_loaded:
            response = "Please upload and process your documents before chatting."
        else:
            with st.spinner("Thinking..."):
                try:
                    import time
                    
                    # Check if we should wait due to rate limits
                    if 'last_request_time' not in st.session_state:
                        st.session_state.last_request_time = 0
                    
                    # Add delay between requests to avoid rate limiting
                    time_since_last = time.time() - st.session_state.last_request_time
                    if time_since_last < 2:  # Wait at least 2 seconds between requests
                        time.sleep(2 - time_since_last)
                    
                    # Invoke the chain
                    response = st.session_state.chain.invoke(user_query)
                    st.session_state.last_request_time = time.time()
                    
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "quota" in error_msg.lower():
                        response = """⚠️ **Rate Limit Exceeded**
                        
You've hit the free tier limits for the Gemini API. Here are your options:

**Immediate Solutions:**
1. **Wait 1 hour** - Free tier quotas reset hourly
2. **Wait until tomorrow** - Daily quotas reset at midnight PST
3. **Use a different Google account** to get a new API key

**Long-term Solutions:**
1. **Upgrade to paid plan** at https://ai.google.dev/pricing
2. **Enable billing** in Google Cloud Console for higher limits

**Free Tier Limits:**
- 15 requests per minute
- 1,500 requests per day
- 1 million tokens per minute

Try again in an hour or upgrade for unlimited usage."""
                    else:
                        response = f"Error: {error_msg}"
        
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Add footer with instructions
st.markdown("---")
st.markdown("""
### How to use TaxCraft:
1. Enter your Google Generative AI API key in the sidebar
2. Upload your tax documents (Knowledge Base and/or Form 16A)
3. Click "Process Documents" and wait for processing to complete
4. Start asking tax-related questions in the chat
""")