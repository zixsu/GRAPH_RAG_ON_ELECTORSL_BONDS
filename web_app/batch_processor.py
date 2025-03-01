import streamlit as st
from model import Model

@st.cache_resource
def load_model():
    return Model('data/final_data.csv')

model = load_model()

st.title("Ask a Question")
if prompt := st.text_input("Enter your question:"):
    response = model.process_query(prompt)
    st.write(f"**Answer:** {response}")
