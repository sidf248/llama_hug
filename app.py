import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

##functions to get rensponse from llama 2 model

def getLLamaresponse(input_text,no_words,blog_style):

    ### LLAMA 2 model

    llm=CTransformers(model='model\llama-2-7b-chat.ggmlv3.q8_0.bin',
                      model_type = 'llama',
                      config={'max_new_tokens':256,
                              'temperature':0.01})
    
    ##Prompt Template

    template="""

        write a blog for {blog_style} job profile for a topic {input_text} within {no_words} words.

        """
    
    prompt =PromptTemplate(input_variables=["blog_style","input_text","no_words"],
                          template = template)

    ##Generate the response from llama model

    response=llm(prompt.format(blog_style=blog_style,input_text=input_text,no_words=no_words))
    print(response)
    return response


st.set_page_config(page_title="Generate Blogs",
                   page_icon='🌏',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Generate Blogs 🌏")

input_text=st.text_input("Enter the Blog Topic")

##creating two more columns for additional 2 fields

col1,col2=st.columns([5,5])

with col1:
    no_words=st.text_input("No of words")

with col2:
    blog_style=st.selectbox('Writing the blog for',
                            ('Researchers','Data Scientist','Common pEOPLE'),index=0)
    
SUBMIT = st.button("Generate")


## Final response

if SUBMIT:
    st.write(getLLamaresponse(input_text,no_words,blog_style))


