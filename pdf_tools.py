__import__('pysqlite3')
import sys,os
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain

os.environ['OPENAI_API_KEY'] = st.secrets.openai_api_key

def get_map_prompt():
    return """Write a summary of this chunk of text that includes the main points and any important details in german. 
{text} """

def get_overall_prompt():
    return """Write a concise summary in german of the following text delimited by triple backquotes. 
Return your response in bullet points which covers the key points of the text. ```{text}```
BULLET POINT SUMMARY:
"""

def get_document_splits(pages, chunk_size=1000):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size * 0.15,
        separators=["\n\n", "\n", "\. ", " ", ""]
    )

    return text_splitter.split_documents(pages)

def accuracy_to_chunk_size(accuracy):
    percentile = accuracy / 100
    return (1 - percentile) * 1000 + percentile * 50;

def get_llm(llm_name):
    return ChatOpenAI(model=llm_name, temperature=0)

def summarize_pdf(chunks, llm, map_prompt_template, combine_prompt_template):
    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
    combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

    chain = load_summarize_chain(
        llm, 
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        return_intermediate_steps=True
    )

    result = chain.invoke(chunks)
    return result["output_text"]

def populate_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

    return vectorstore

def on_file_change():
    st.session_state.clear()

st.title("PDF Tools")
picked_file = st.file_uploader("Upload a file", type="pdf", on_change=on_file_change)

if(picked_file is not None):
    with open(picked_file.name, mode='wb') as w:
        w.write(picked_file.getvalue())

    loader = PyPDFLoader(picked_file.name)
    pages = loader.load()
    page_min, page_max = st.slider("Which pages should be used?", 0, len(pages), (0, len(pages)))
    llm_name = st.selectbox("Which LLM do you want to use?", ("gpt-3.5-turbo", "gpt-4o"))
    tab_qa, tab_summary = st.tabs(["Question Answering", "Summary"])

    with tab_summary:
        with st.form("my_form"):
            st.warning("""This method lets the entire document be processed by the LLM and is therefore pretty slow (and expensive)
                    \nQuestion Answering with high accuracy can also create good summaries!""")
            st.text("""The PDF will be split in small processable chunks and then summarized. \nPrompts can be customized for both tasks below.""")
            text_prompt_map = st.text_area("Chunk Prompt:", get_map_prompt())
            text_prompt_combine = st.text_area("Summary Prompt:", get_overall_prompt())

            submitted = st.form_submit_button("Submit")

            if(submitted):
                llm = get_llm(llm_name)
                pages = pages[page_min:page_max]
                chunks = get_document_splits(pages, 1000)
                response = summarize_pdf(chunks, llm, text_prompt_map, text_prompt_combine)

                st.info(response)

    with tab_qa:
        with st.form("index"):
            accuracy = st.slider("Accuracy", 0, 100, 10)
            submitted = st.form_submit_button("Index Document")

        if(submitted):
            pages = pages[page_min:page_max]
            chunks = get_document_splits(pages, accuracy_to_chunk_size(accuracy))
            store = populate_vectorstore(pages)

            st.session_state["db"] = store

        if 'db' in st.session_state:
            with st.form("quesiton"):
                question = st.text_area('Ask Quesiton:', '')
                submitted = st.form_submit_button("Submit")

                if(submitted):
                    store = st.session_state.db
                    llm = get_llm(llm_name)
                    retrieval = store.similarity_search(query=question, k=3)
                    chain = load_qa_chain(llm=llm, chain_type="map_reduce")

                    response = chain.run(input_documents=retrieval, question=question)

                    st.write(response)
