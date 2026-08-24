import os

import chromadb
from ollama import chat
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "sentence-transformers/all-miniLM-L6-v2"
)


client = chromadb.PersistentClient(
    path="./chroma_db"
)

print("Available collections:")
print(client.list_collections())

collections = client.get_collection(
    name="documents",
    embedding_function=None
)

while True:
    question=input("\nYou: ")

    if question.strip().lower() == "exit":
        print("chat closed")
        break 

    question_embedding=embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results=collections.query(
        query_embeddings=[question_embedding],
        n_results=4
    )

    retrieved_chunks = results["documents"][0]

    context = "\n\n".join(retrieved_chunks)

    prompt=f"""You are a document question-answering assistant.

Answer the question using only the provided context.

If the answer cannot be found in the context,
say that you do not have enough information.

CONTEXT:
{context}

QUESTION:
{question}
"""


    response = chat(
        model="qwen3:4b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    print("\nAssistant:")
    print(response.message.content)