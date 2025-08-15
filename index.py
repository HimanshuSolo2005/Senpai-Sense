import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

load_dotenv()

loader = TextLoader("notes.md")  
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
vectorstore.persist()

print("✅ Documents embedded and stored in ChromaDB.")
