from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# 1. Load all 3 PDFs with metadata
pdf_files = [
    {"path": "MTSK_Installation_guide_R02.pdf", "type": "installation"},
    {"path": "02_Create the robot folder.pdf", "type": "robotics"},
    {"path": "04_Machine Tool Builder.pdf", "type": "machine_tool"},
]

all_chunks = []
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

for pdf in pdf_files:
    loader = PyPDFLoader(pdf["path"])
    pages = loader.load()
    for page in pages:
        page.metadata["doc_type"] = pdf["type"]
        page.metadata["doc_name"] = pdf["path"]
    chunks = splitter.split_documents(pages)
    all_chunks.extend(chunks)
    print(f"Loaded {pdf['path']} — {len(pages)} pages, {len(chunks)} chunks")

print(f"\nTotal chunks across all documents: {len(all_chunks)}")

# 2. Embed and store in persistent ChromaDB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=all_chunks,
    embedding=embeddings,
    persist_directory="./chroma_db_multi"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 3. Build chain
prompt = ChatPromptTemplate.from_template("""
You are a manufacturing and CNC engineering assistant.
Answer using only the context below. Be precise and concise.
If the answer spans multiple documents, mention which document it came from.
If the answer is not in the context, say so clearly.

Context: {context}

Question: {question}
""")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def format_docs(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("doc_name", "unknown")
        formatted.append(f"[From: {source}]\n{doc.page_content}")
    return "\n\n".join(formatted)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 4. Ask cross-document questions
def ask(question):
    answer = chain.invoke(question)
    print(f"\nQ: {question}")
    print(f"A: {answer}")

ask("What are the installation steps?")
ask("How do I create a robot folder?")
ask("What is the Machine Tool Builder used for?")
ask("What software tools are mentioned across all documents?")