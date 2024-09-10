import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Function to get response from Llama 2 model
def getLLamaresponse(input_text, no_words, blog_style, retriever=None):
    # Initialize the LLAMA 2 model
    llm = CTransformers(
        model='D:/git/1/llama_hug-main/model/llama-2-7b-chat.ggmlv3.q8_0.bin',  # Ensure path correctness
        model_type='llama',
        config={
            'max_new_tokens': 256,
            'temperature': 0.01
        }
    )

    # Prompt Template
    template = """
    Write interview questions for {blog_style} job profile for a topic {input_text} within {no_words} words.
    """

    prompt = PromptTemplate(
        input_variables=["blog_style", "input_text", "no_words"],
        template=template
    )

    if retriever:
        # Create a RetrievalQA chain if retriever is available
        chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
        response = chain.run(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    else:
        # Generate the response from the Llama model without retrieval
        response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    
    return response

# Function to upload and process documents
def process_documents(files):
    documents = []
    for file in files:
        if file.type == "application/pdf":
            loader = PyPDFLoader(file)
        else:
            loader = TextLoader(file)
        documents.extend(loader.load())

    # Split documents into smaller chunks for indexing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(texts, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Adjust 'k' as needed

    return retriever

# Streamlit UI setup
st.set_page_config(
    page_title="Generate Blogs",
    page_icon='🌏',
    layout='centered',
    initial_sidebar_state='collapsed'
)

st.header("Generate Blogs 🌏")

# Upload documents
uploaded_files = st.file_uploader("Upload Documents (PDF, TXT)", type=['pdf', 'txt'], accept_multiple_files=True)

retriever = None
if uploaded_files:
    retriever = process_documents(uploaded_files)

input_text = st.text_input("Enter the Blog Topic")

# Creating two more columns for additional 2 fields
col1, col2 = st.columns([5, 5])

with col1:
    try:
        no_words = int(st.text_input("No of words", value="250"))  # Set a default value
    except ValueError:
        no_words = 250  # Default in case of invalid input

with col2:
    blog_style = st.selectbox(
        'Writing the blog for',
        ('Researchers', 'Data Scientist', 'Common People'), index=0
    )

SUBMIT = st.button("Generate")

# Final response
if SUBMIT:
    response = getLLamaresponse(input_text, no_words, blog_style, retriever)
    st.write(response)  # Use st.write to display the response properly
