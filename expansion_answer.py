from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI
import os
import umap

from helper_utils import word_wrap, project_embeddings
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

reader = PdfReader("data/me.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

pdf_texts = [text for text in pdf_texts if text]

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter
)

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". "],
    chunk_size=200,
    chunk_overlap=80
)

character_split_texts = character_splitter.split_text(" ".join(pdf_texts))

token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=50
)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

embedding_function = SentenceTransformerEmbeddingFunction()

chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection("scacchi", 
                                embedding_function=embedding_function)

ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(
    ids=ids,
    documents=token_split_texts
)


def augment_query_generated(query, model="gpt-3.5-turbo"):
    prompt = """Sei un assistente che fornisce informazioni su persone private.
   Fornisci una risposta esemplificativa alla domanda data, come se fosse tratta da un documento personale."""
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content

original_query = "Chi è davide michelon?"
hypothetical_answer = augment_query_generated(original_query)

print(f"hypothetical_answer: {hypothetical_answer}")
joint_query = f"{original_query} {hypothetical_answer}"

results = chroma_collection.query(
    query_texts=joint_query, n_results=10, include=["documents", "embeddings"]
)

retrieved_documents = results["documents"][0]

def generate_response_from_retrieved_docs(query, documents, model="gpt-3.5-turbo"):
    """
    Genera una risposta utilizzando GPT basata sui documenti recuperati.
    
    Args:
        query (str): La query originale dell'utente.
        documents (list): I documenti recuperati dalla ricerca vettoriale.
        model (str): Il modello GPT da utilizzare.
        
    Returns:
        str: La risposta generata.
    """
    # Prepara il contesto con i documenti recuperati
    context = "\n\n".join(documents)
    
    # Crea il prompt per GPT
    prompt = f"""Sei un assistente che fornisce informazioni precise basate sui documenti forniti.
    Utilizza SOLO le informazioni contenute nei seguenti documenti per rispondere alla domanda.
    Se non puoi rispondere in base ai documenti forniti, dì onestamente che non hai abbastanza informazioni.
    
    Documenti:
    {context}
    
    Domanda: {query}
    
    Risposta:"""
    
    # Chiama l'API di OpenAI
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Sei un assistente che fornisce risposte basate solo sui documenti forniti."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3  # Temperatura bassa per risposte più precise
    )
    
    return response.choices[0].message.content

# Genera la risposta finale utilizzando i documenti recuperati
final_answer = generate_response_from_retrieved_docs(original_query, retrieved_documents)

print("\n=== RISPOSTA FINALE ===\n")
print(word_wrap(final_answer))


embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=42, transform_seed=42).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)


retrieved_embeddings = results["embeddings"][0]
original_query_embedding = embedding_function([original_query])
augmented_query_embedding = embedding_function([joint_query])

projected_original_query_embedding = project_embeddings(
    original_query_embedding, umap_transform
)
projected_augmented_query_embedding = project_embeddings(
    augmented_query_embedding, umap_transform
)
projected_retrieved_embeddings = project_embeddings(
    retrieved_embeddings, umap_transform
)

import matplotlib.pyplot as plt

# Plot the projected query and retrieved documents in the embedding space
plt.figure()

plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="gray",
)
plt.scatter(
    projected_retrieved_embeddings[:, 0],
    projected_retrieved_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
)
plt.scatter(
    projected_original_query_embedding[:, 0],
    projected_original_query_embedding[:, 1],
    s=150,
    marker="X",
    color="r",
)
plt.scatter(
    projected_augmented_query_embedding[:, 0],
    projected_augmented_query_embedding[:, 1],
    s=150,
    marker="X",
    color="orange",
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f"{original_query}")
plt.axis("off")
plt.show()  # display the plot