import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma


load_dotenv()

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

llm = ChatGroq(
    model="llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

while True:
    query = input("\nAsk: ")
    if query.lower() in ["exit", "quit"]:
        break
    answer = qa.run(query)
    print("Answer:", answer)
