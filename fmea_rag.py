from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from fmea_engine import generate_fmea
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "./fmea_knowledge_db"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_process_documents(pdf_paths):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50)
    all_chunks = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        pages = loader.load()
        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    vectorstore.add_documents(all_chunks)
    print(f"Ingested {len(all_chunks)} chunks")

def retrieve_process_context(process_name, process_steps):
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    query = f"{process_name} {' '.join(process_steps)} failure modes quality"
    docs = vectorstore.similarity_search(query, k=5)
    context = "\n".join([d.page_content for d in docs])
    return context

def generate_fmea_with_context(process_name, process_steps,
                                machine, material, quality_req,
                                tolerances, pdf_paths=[]):
    if pdf_paths:
        ingest_process_documents(pdf_paths)
    context = retrieve_process_context(process_name, process_steps)
    print(f"Retrieved context: {len(context)} characters")
    return generate_fmea(
        process_name, process_steps,
        machine, material, quality_req,
        tolerances, context
    )

if __name__ == "__main__":
    result = generate_fmea_with_context(
        process_name="CNC Milling — Aluminium Housing",
        process_steps=["Material loading", "Clamping",
                       "Rough milling", "Finish milling", "Inspection"],
        machine="DMG MORI DMU 50",
        material="Aluminium 6061-T6",
        quality_req=["Surface finish Ra 0.8", "Tolerance +/-0.05mm"],
        tolerances=["Bore diameter +/-0.02mm", "Flatness 0.05mm",
                    "Perpendicularity 0.03mm"],
        pdf_paths=["MTSK_Installation_guide_R02.pdf"]
    )
    import json
    print(json.dumps(result, indent=2))