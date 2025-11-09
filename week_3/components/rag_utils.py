from __future__ import annotations
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def chunk_docs(raw_texts: List[str], sources: List[str] | None = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = []
    sources = sources or [f"doc_{i}" for i in range(len(raw_texts))]
    for src, txt in zip(sources, raw_texts):
        if not txt:
            continue
        for chunk in splitter.split_text(txt):
            docs.append(Document(page_content=chunk, metadata={"source": src}))
    return docs


def build_chroma(docs: List[Document], collection_name: str = "voice_rag"):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma.from_documents(docs, embedding=embeddings, collection_name=collection_name)
    return vectordb

def answer_with_rag(vectordb, llm_call, query: str, k: int = 4) -> str:
    retrieved_docs = vectordb.similarity_search(query, k=k)
    context = "\n\n".join([f"[Source: {d.metadata.get('source', '?')}]\n{d.page_content}\n" for d in retrieved_docs])
    prompt = f"""
You are a helpful assistant. Use ONLY the context to answer.
If the answer isn't in the context, say you don't have enough information.


Context:\n{context}


Question: {query}
Answer concisely:
"""
    return llm_call(prompt)