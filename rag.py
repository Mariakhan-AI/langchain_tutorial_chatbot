import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain_groq import ChatGroq
from langchain import hub
from langchain.agents import create_openai_tools_agent, AgentExecutor
import time

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Advanced RAG Q/A Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1f77b4;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .sidebar-content {
        padding: 1rem;
    }
    .stAlert > div {
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'agent_executor' not in st.session_state:
    st.session_state.agent_executor = None
if 'initialization_done' not in st.session_state:
    st.session_state.initialization_done = False

@st.cache_resource
def initialize_tools_and_agent():
    """Initialize tools and agent with caching for better performance"""
    try:
        # Set up API key
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("GROQ_API_KEY not found in environment variables. Please check your .env file.")
            return None
        
        os.environ["GROQ_API_KEY"] = groq_api_key
        
        # Initialize LLM
        llm = ChatGroq(model_name="llama3-70b-8192", temperature=0)
        
        # Initialize Wikipedia tool
        api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
        wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
        
        # Initialize ArXiv tool
        arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
        arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
        
        # Load and process Hugging Face blog
        with st.spinner("Loading Hugging Face blog content..."):
            loader = WebBaseLoader("https://huggingface.co/blog")
            docs = loader.load()
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            documents = text_splitter.split_documents(docs)
            
            # Create vector database
            embeddings = HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-v2'
            )
            vectordb = FAISS.from_documents(documents, embeddings)
            
            # Create retriever (Fix: call the method with parentheses)
            retriever = vectordb.as_retriever()
            
            # Create retrieval tool
            retrieval_tool = create_retriever_tool(
                retriever, 
                "huggingface_blog", 
                "Search for information from Hugging Face blog posts and articles"
            )
        
        # Combine all tools
        tools = [wiki, arxiv, retrieval_tool]
        
        # Load prompt template
        prompt = hub.pull("hwchase17/openai-functions-agent")
        
        # Create agent
        agent = create_openai_tools_agent(llm, tools, prompt)
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )
        
        return agent_executor, tools
        
    except Exception as e:
        st.error(f"Error initializing tools and agent: {str(e)}")
        return None, None

def main():
    # Header
    st.markdown("<h1 class='main-header'>🤖 Advanced RAG Q/A Chatbot with Multiple Data Sources (Wikipedia, ArXiv and Hugging Face)</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
        
        st.markdown("### 📚 Available Data Sources")
        st.info("📖 **Wikipedia**: General knowledge search")
        st.info("🔬 **ArXiv**: Academic paper search")
        st.info("🤗 **Hugging Face Blog**: Latest AI/ML insights")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Initialize tools if not done
        if not st.session_state.initialization_done:
            with st.spinner("🚀 Initializing Chatbot..."):
                result = initialize_tools_and_agent()
                if result and result[0] is not None:
                    st.session_state.agent_executor, tools = result
                    st.session_state.initialization_done = True
                    st.success("✅ Chatbot initialized successfully!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Failed to initialize Chatbot")
                    st.stop()
        
        # Chat interface
        if st.session_state.agent_executor:
            # Display chat history
            chat_container = st.container()
            with chat_container:
                for i, (role, message) in enumerate(st.session_state.chat_history):
                    if role == "user":
                        st.markdown(f"""
                        <div class="chat-message user-message">
                            <strong>🧑 You:</strong><br>{message}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-message assistant-message">
                            <strong>🤖 Assistant:</strong><br>{message}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Chat input
            user_question = st.chat_input("Ask me anything from Wikipedia, ArXiv, or Hugging Face blog...")
            
            if user_question:
                # Add user message to history
                st.session_state.chat_history.append(("user", user_question))
                
                # Show thinking spinner
                with st.spinner("🤔 Thinking and searching..."):
                    try:
                        # Get response from agent
                        response = st.session_state.agent_executor.invoke({
                            "input": user_question
                        })
                        
                        # Add assistant response to history
                        assistant_response = response.get("output", "Sorry, I couldn't generate a response.")
                        st.session_state.chat_history.append(("assistant", assistant_response))
                        
                    except Exception as e:
                        error_message = f"Sorry, I encountered an error: {str(e)}"
                        st.session_state.chat_history.append(("assistant", error_message))
                
                # Rerun to show new messages
                st.rerun()
        
        else:
            st.warning("⚠️ Chatbot not initialized. Please check your configuration.")
    
    with col2:
        # Quick actions and examples
        st.markdown("### 💡 Try These Examples")
        
        example_questions = [
            "What is Hugging Face?",
            "Latest trends in transformer models",
            "How does BERT work?",
            "Recent papers on large language models",
            "What is RAG in AI?",
            "Explain attention mechanism"
        ]
        
        for question in example_questions:
            if st.button(f"❓ {question}", key=f"example_{hash(question)}", use_container_width=True):
                # Simulate clicking the example
                st.session_state.chat_history.append(("user", question))
                
                with st.spinner("🤔 Thinking and searching..."):
                    try:
                        if st.session_state.agent_executor:
                            response = st.session_state.agent_executor.invoke({
                                "input": question
                            })
                            assistant_response = response.get("output", "Sorry, I couldn't generate a response.")
                            st.session_state.chat_history.append(("assistant", assistant_response))
                    except Exception as e:
                        error_message = f"Sorry, I encountered an error: {str(e)}"
                        st.session_state.chat_history.append(("assistant", error_message))
                
                st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🤖 Powered by Groq (Llama 3), Hugging Face, Wikipedia & ArXiv</p>
        <p>Built with ❤️ using Streamlit & LangChain</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()