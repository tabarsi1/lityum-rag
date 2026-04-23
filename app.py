import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile, os
from dotenv import load_dotenv
import os
os.environ["OPENAI_API_KEY"] = "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"

load_dotenv()

st.set_page_config(page_title="Lityum Engineering Assistant", layout="wide")
st.title("Lityum Engineering")
st.caption("Manufacturing Document Assistant — powered by AI")
st.divider()

if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Upload Documents")
    st.caption("Upload your manufacturing PDFs to get started")
    uploaded = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )
    if uploaded and st.button("Process Documents", type="primary"):
        with st.spinner("Processing your documents..."):
            all_chunks = []
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            for f in uploaded:
                tmp = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf")
                tmp.write(f.read())
                tmp.close()
                loader = PyPDFLoader(tmp.name)
                pages = loader.load()
                for page in pages:
                    page.metadata["doc_name"] = f.name
                chunks = splitter.split_documents(pages)
                all_chunks.extend(chunks)
                try:
                    os.unlink(tmp.name)
                except:
                    pass

            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            vectorstore = Chroma.from_documents(all_chunks, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

            prompt = ChatPromptTemplate.from_template("""
You are a manufacturing engineering assistant for Lityum Engineering.
Answer using only the context below. Be precise and concise.
Always mention which document the answer comes from.
If the answer is not in the context, say so clearly.

Context: {context}

Question: {question}
""")
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

            def format_docs(docs):
                formatted = []
                for doc in docs:
                    source = doc.metadata.get("doc_name", "unknown")
                    formatted.append(
                        f"[From: {source}]\n{doc.page_content}")
                return "\n\n".join(formatted)

            st.session_state.chain = (
                {"context": retriever | format_docs,
                 "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            st.success(
                f"Processed {len(all_chunks)} chunks "
                f"from {len(uploaded)} documents")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt_input := st.chat_input("Ask anything about your documents..."):
    if not st.session_state.chain:
        st.error("Please upload documents first using the sidebar.")
    else:
        st.session_state.messages.append(
            {"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.write(prompt_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.chain.invoke(prompt_input)
                st.write(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer})