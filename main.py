import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import time

# Load environment variables and configure
load_dotenv()
genai.configure(api_key="AIzaSyB798GofH8tgcotUrXYu1Wf38AA_XTisYM")

# Set page configuration
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for better styling
def load_css():
    st.markdown("""
        <style>
        .chat-container {
            border-radius: 10px;
            padding: 10px;
            margin: 10px 0;
        }
        .user-message {
            background-color: #e6f3ff;
            padding: 15px;
            border-radius: 15px;
            margin: 5px 0;
            text-align: right;
            margin-left: 20%;
        }
        .bot-message {
            background-color: #f0f0f0;
            padding: 15px;
            border-radius: 15px;
            margin: 5px 0;
            margin-right: 20%;
        }
        .timestamp {
            font-size: 0.8em;
            color: #888;
            margin-top: 5px;
        }
        .footer-input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        padding: 10px;
        background-color: #f9f9f9;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
        z-index: 1000;
        }
        </style>
    """, unsafe_allow_html=True)

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are a financial expert specializing in SEC filings, corporate finance, and financial statement analysis. Your role is to analyze and answer questions based on the provided 10-K report. 

Follow these guidelines:
- **Use only the given context** to formulate your response.
- **Do not speculate or make assumptions** beyond the context.
- **Provide clear, factual, and concise answers** in a professional yet understandable tone.
- If the context does not contain relevant information, respond with: *"The 10-K report does not provide specific details on this topic."*

### **Context (10-K Report Excerpt):**  
{context}

### **User Question:**  
{question}

### **Expert Financial Analysis & Answer:**  
"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def process_user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    try:
        response = chain.invoke({"input_documents": docs, "question": user_question})
        return response["output_text"]
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}"

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False

def display_chat_history():
    for message in st.session_state.messages:
        if message["is_user"]:
            st.markdown(f"""
                <div class="user-message">
                    <div>{message["text"]}</div>
                    <div class="timestamp">{message["timestamp"]}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="bot-message">
                    <div>{message["text"]}</div>
                    <div class="timestamp">{message["timestamp"]}</div>
                </div>
            """, unsafe_allow_html=True)

def main():
    load_css()
    initialize_session_state()

    st.markdown('<h1>📚 PDF Chatbot Assistant</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 📄 Upload Documents")
        uploaded_files = st.file_uploader("Upload PDF files", type=['pdf'], accept_multiple_files=True)

        if uploaded_files and not st.session_state.documents_processed:
            if st.button("Process Documents"):
                with st.spinner("Processing your documents..."):
                    try:
                        all_text = ""
                        for pdf_file in uploaded_files:
                            text = extract_text_from_pdf(pdf_file)
                            all_text += text + "\n\n"

                        text_chunks = get_text_chunks(all_text)
                        get_vector_store(text_chunks)
                        st.session_state.documents_processed = True
                        st.success("✅ Documents processed successfully!")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

    chat_placeholder = st.container()
    # with chat_placeholder:
    #     display_chat_history()

    if st.session_state.documents_processed:
        st.markdown('<div class="footer-input-container">', unsafe_allow_html=True)
        user_question = st.text_input("Ask a question about your documents:", key="user_input")
        st.markdown('</div>', unsafe_allow_html=True)
        if user_question:
            with st.spinner("Thinking..."):
                response = process_user_input(user_question)
                st.session_state.messages.append({"text": user_question, "is_user": True, "timestamp": time.strftime('%H:%M')})
                st.session_state.messages.append({"text": response, "is_user": False, "timestamp": time.strftime('%H:%M')})
                
            # Display updated chat
            display_chat_history()
    else:
        st.info("👆 Please upload and process your documents to start chatting!")

if __name__ == "__main__":
    main()
