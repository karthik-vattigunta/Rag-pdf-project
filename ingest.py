import os
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

PDF_PATH="data/sample.pdf"

def read_pdf(path):
    reader = PdfReader(path)

    pages=[]

    for page_number,page in enumerate(reader.pages,start=1):
        text=page.extract_text() or ""

        pages.append({
            "page":page_number,
            "text":text
        })

    return pages


def chunk_text(text,chunk_size=200,overlap=20):
    words=text.split()

    chunks=[]
    start=0

    while start < len(words):
        end=start+chunk_size

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start+=chunk_size-overlap

    return chunks
# loads a model that converts text into the embeddings
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)
# create a chroma vector data base and saved locally in the chroma_db folder
client=chromadb.PersistentClient(
    path="./chroma_db"
)
# gets or create a collection called documents and tells chroma that you will provide the embeddings your self
collection = client.get_or_create_collection(
    name="documents",
    embedding_function=None

)

pages = read_pdf(PDF_PATH)

documents = []
metadatas=[]
ids=[]

chunk_id=0

for page in pages:
    chunks = chunk_text(page["text"])

    for chunk in chunks:

        documents.append(chunk)

        metadatas.append({
            "page": page["page"],
            "source": PDF_PATH
        })

        ids.append(str(chunk_id))

        chunk_id += 1

embeddings=embedding_model.encode(
    documents,normalize_embeddings=True
).tolist()

collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings
)

print(f"Stored {len(documents)} chunks.")