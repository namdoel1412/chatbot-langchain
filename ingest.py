"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle
import os

from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
os.environ["OPENAI_API_KEY"] = "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"

def ingest_docs():
    """Get documents from web pages."""
    loader = ReadTheDocsLoader("./docs/TentenDoc.pdf")
    raw_documents = loader.load()
    print(raw_documents)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    ingest_docs()
