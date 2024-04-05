import streamlit as st
from langchain.llms import OpenAI
#from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains import LLMChain,SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper #install it using pip install wikipedia
import os

#keeping API KEY in env file and load it from there
load_dotenv()
KEY = os.getenv("OPENAI_API_KEY")

st.title("Youtube script generator title using lang chain..")

#take input from user
userinput=st.text_input(" Plug in your prompt here.. ")

#use of prompt template
title_template = PromptTemplate(input_variables=['topic','wikipedia_research'],template='write me a youtube video about {userinput}, using this wiki resource {wikipedia_research}')
script_template = PromptTemplate(input_variables=['topic'],template='write me a youtube video script based on {userinput}')    

#Memory
title_memory=ConversationBufferMemory(input_key='topic',memory_key='chat_history')
script_memory=ConversationBufferMemory(input_key='title',memory_key='chat_history')

#LLM
llm= OpenAI(temperature=0.9)
#use of sequential chain
title_chain=LLMChain(llm=llm, prompt=title_template,verbose=True,output_key='title',memory=title_memory)
script_chain=LLMChain(llm=llm, prompt=script_template,verbose=True,output_key='script',memory=script_memory)



wiki = WikipediaAPIWrapper()
if userinput:
  title =title_chain.run(userinput)
  wiki_research =wiki.run(userinput)
  script = script_chain.run(title=title,wikipedia_research=wiki_research)
  st.write(title)
  st.write(script)
  with st.expander('Title History'):
        st.info(title_memory.buffer)
  with st.expander('Script History'):
        st.info(script_memory.buffer)
  with st.expander('Wikipedia Research'):
        st.info(wiki_research)