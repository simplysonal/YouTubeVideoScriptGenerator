import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains import LLMChain,SequentialChain
import os

#keeping API KEY in env file and load it from there
load_dotenv()
KEY = os.getenv("OPENAI_API_KEY")

st.title("Youtube script generator title using lang chain..")

#take input from user
userinput=st.text_input(" Plug in your prompt here.. ")

#use of prompt template
title_template = PromptTemplate(input_variables=['topic'],template='write me a youtube video about {topic}')
script_template = PromptTemplate(input_variables=['topic'],template='write me a youtube video script based on {topic}')    


#LLM
llm= OpenAI(temperature=0.9)
#use of sequential chain
title_chain=LLMChain(llm=llm, prompt=title_template,verbose=True,output_key='title')
script_chain=LLMChain(llm=llm, prompt=script_template,verbose=True,output_key='script')

sequential_chain =SequentialChain(chains=[title_chain,script_chain],input_variables=['topic'],
                                        output_variables=['title','script'],verbose =True)

if userinput:
    #response=llm(userinput)
    #response=title_chain.run(topic=userinput)
    #response = sequential_chain.run({'topic':userinput})

    #writing response
    response = sequential_chain({'topic':userinput})
    st.write(response['title'])
    st.write(response['script'])