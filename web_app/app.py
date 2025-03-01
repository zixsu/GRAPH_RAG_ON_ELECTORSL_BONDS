import streamlit as st
from model import Model  # Ensure the correct import

# Initialize RAG system
@st.cache_resource
def load_rag():
    return Model('data/final_data.csv') 

rag = load_rag()

st.title("Political Bond Query System")

# User input
question = st.text_input("Enter your query:")

if st.button("Get Answer"):
    if question.strip():
        with st.spinner("Processing..."):
            answer = rag.process_query(question.strip())
            st.write("**Answer:**", answer)
    else:
        st.warning("Please enter a valid query.")
