import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains import LLMChain
import os

load_dotenv()
KEY = os.getenv("OPENAI_API_KEY")

st.title("Youtube script generator title using lang chain..")

userinput=st.text_input(" Plug in your prompt here.. ")

#use of prompt template
title_template = PromptTemplate(input_variables=['topic'],template='write me a youtube video about {topic}')


#lLM

llm= OpenAI(temperature=0.9)
title_chain=LLMChain(llm=llm, prompt=title_template,verbose=True)

if userinput:
    #response=llm(userinput)
    response=title_chain.run(topic=userinput)
    st.write(response)