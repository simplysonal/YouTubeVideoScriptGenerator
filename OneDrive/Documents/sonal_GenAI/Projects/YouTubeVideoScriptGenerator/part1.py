import streamlit as st
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
KEY = os.getenv("OPENAI_API_KEY")

st.title("Youtube script generator title using lang chain..")

prompt=st.text_input(" Plug in your prompt here.. ")

#lLM
llm= OpenAI(temperature=0.9)

if prompt:
    response=llm(prompt)
    st.write(response)