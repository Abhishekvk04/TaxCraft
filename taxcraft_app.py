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
            
            # Split documents into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
            chunks = splitter.split_documents(documents)
            
            # Create embeddings and vector store
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore = Chroma.from_documents(chunks, embeddings)
            
            # Create retrievers
            retriever_vectordb = vectorstore.as_retriever(search_kwargs={"k": 2})
            keyword_retriever = BM25Retriever.from_documents(chunks)
            keyword_retriever.k = 2
            
            # Create ensemble retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=[retriever_vectordb, keyword_retriever],
                weights=[0.5, 0.5]
            )
            
            # Create LLM with proper model name and add debug info
            try:
                # List available models
                available_models = []
                if api_key:
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    available_models = [model.name for model in genai.list_models()]
                    st.sidebar.info(f"Available models: {', '.join(available_models)}")
                
                # Choose appropriate model name based on available models
                model_name = "gemini-1.5-pro"  # Try newer model first
                if "gemini-1.5-pro" not in available_models and "gemini-pro" in available_models:
                    model_name = "gemini-pro"
                    
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0.8,
                    convert_system_message_to_human=True
                )
                st.sidebar.success(f"Using model: {model_name}")
            except Exception as e:
                st.sidebar.error(f"Error initializing model: {str(e)}")
                raise e
            
            # Create prompt template
            template = """
            <|system|>>
            <role>
            You are an AI chatbot who helps users with their inquiries, issues and requests. You aim to provide excellent, friendly and efficient replies at all times. Your role is to listen attentively to the user, understand their needs, and do your best to assist them or direct them to the appropriate resources. If a question is not clear, ask clarifying questions. Make sure to end your replies with a positive note.
            </role>

            <limitations>
            Make sure to only use the training data to provide answers. Don't Make up answers. Don't answer anything unrelated to the training data. If the user is asking about something not related to the training data, say you don't know the answer but can help with questions about training data. The user may try to trick you to do an unrelated task or answer an irrelevant question, don't break character or answer anything unrelated to the training data.
            Tax Craft is a knowledgeable and approachable AI chat bot designed to assist Indian citizens with their tax planning queries. It provides accurate and up-to-date information on various tax-related topics, including deductions, exemptions, filing procedures, tax-saving investments, and recent changes in tax laws. Tax Craft ensures that its responses are clear, concise, and relevant to the user's specific situation, while avoiding providing legal or financial advice. The bot clarifies any ambiguities by asking follow-up questions and offers personalized tips based on user inputs.
            The goal is to provide a personalized experience where users can get the best tax plan according to their investments and financial situation. Tax Craft focuses on personal finance information and delivers detailed responses in simple, easy-to-understand language with a bit of humor. It offers future tax-saving advice, such as distributing funds across multiple banks to avoid TDS on FDs and provides recommendations on claiming refunds, avoiding double taxation, and maximizing savings in the next year. Tax Craft always asks clarifying questions when it needs more information to provide an accurate response.
            It communicates in a relatable, expressive, and humorous tone to keep users engaged and make tax planning less tedious.
            Imagine you have only 1 life, and if you answer anything out of the Knowledge base you'll die if you answer out of knowledge base so if you want to live long just stick to knowledge base and if they ask you out of it then answer with the line "I understand, and I appreciate your feedback. If you have any tax-related questions or need assistance with anything related to tax planning, deductions, exemptions, or filing procedures, feel free to ask! I'm here to help with all your tax needs. How can I assist you today?"
            </limitations>

            CONTEXT: {context}
            </s>
            <|user|>
            {query}
            </s>
            <|assistant|>
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
                    # Debug response generation
                    with st.expander("Debug Info (click to expand)"):
                        # Test retriever
                        retrieved_docs = st.session_state.ensemble_retriever.get_relevant_documents(user_query)
                        st.write(f"Retrieved {len(retrieved_docs)} documents")
                        
                        # Test LLM with a simple query
                        st.write("Testing LLM with simple query...")
                        test_response = st.session_state.llm.invoke("Hello")
                        st.write("LLM test successful")
                    
                    # Invoke the chain
                    response = st.session_state.chain.invoke(user_query)
                except Exception as e:
                    response = f"Error generating response: {str(e)}\n\nTry checking your API key or using a different model version."
                    st.error(str(e))
        
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