from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set keys from environment
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Set Streamlit page config
st.set_page_config(
    page_title="AI Dermatologist Assistant",
    page_icon="🧴",
    layout="centered"
)

# Title and intro
st.markdown(
    "<h1 style='text-align: center; color: #6A0DAD;'>🧴 Your AI Dermatologist</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: #333;'>Powered by LangChain + Groq + LLaMA 3</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and expert dermatologist. Answer skin-related questions with professional advice."),
    ("user", "Question: {question}")
])

# LLM setup
llm = ChatGroq(model="llama3-8b-8192")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# User input
st.subheader("💬 Describe your skin concern")
input_text = st.text_input("e.g., I have acne and oily skin. What should I use?")

# Generate response
if input_text:
    with st.spinner("Analyzing your skin problem..."):
        response = chain.invoke({"question": input_text})
    st.markdown("### 🧾 Recommended Advice:")
    st.success(response)

# Footer
st.markdown("---")
st.markdown(
    "<small style='color: gray;'>Built with ❤️ using Streamlit, LangChain, and LLaMA 3 via Groq API.</small>",
    unsafe_allow_html=True
)
