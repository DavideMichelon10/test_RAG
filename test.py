import os
import csv
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()


openai_api_key = os.getenv("OPENAI_API_KEY")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key,
    model_name="text-embedding-3-small"
)

chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=openai_ef
)

client_openai = OpenAI(api_key=openai_api_key)

# Funzione per caricare i dati dal file CSV
def load_data_from_csv(file_path):
    print("==== Caricamento dati dal file CSV ====")
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append({
                "specialita": row["Specialità"],
                "dottore": row["Dottore"]
            })
    return data

def get_openai_embeddings(text):
    return client_openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

def analizza_richiesta(testo):
    """Analizza il testo dell'utente per estrarre specialità e dottore"""
    # Usiamo OpenAI per analizzare la richiesta dell'utente
    prompt = (
        "Estrai la specialità medica e/o il nome del dottore dal seguente testo. "
        "Se non sono presenti, restituisci stringa vuota. Formato: 'Specialità: X, Dottore: Y'. "
        f"Testo: '{testo}'"
    )
    
    response = client_openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Sei un assistente che estrae informazioni."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    
    result = response.choices[0].message.content
    
    # Estrai specialità e dottore dalla risposta
    specialita = ""
    dottore = ""
    
    if "Specialità:" in result:
        specialita_part = result.split("Specialità:")[1].split(",")[0].strip()
        if specialita_part and specialita_part.lower() != "nessuna" and specialita_part != "X":
            specialita = specialita_part
            
    if "Dottore:" in result:
        dottore_part = result.split("Dottore:")[1].strip()
        if dottore_part and dottore_part.lower() != "nessuno" and dottore_part != "Y":
            dottore = dottore_part
            
    return specialita, dottore

def trova_specialita_simili(query, n=3):
    """Trova le specialità più simili al testo della query"""
    embedding = get_openai_embeddings(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n
    )
    
    if results["metadatas"]:
        return set([item["specialita"] for sublist in results["metadatas"] for item in sublist])
    return set()

def trova_dottori_simili(query, n=3):
    """Trova i dottori più simili al testo della query"""
    embedding = get_openai_embeddings(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n
    )
    
    if results["metadatas"]:
        return set([item["dottore"] for sublist in results["metadatas"] for item in sublist])
    return set()

def interactive_chat():
    print("===== Benvenuto al sistema di prenotazione visite =====")
    print("Posso aiutarti a trovare la specialità medica e il dottore giusto per te.")
    
    specialita = None
    dottore = None
    tentativi = 0
    
    while not specialita or (not dottore and tentativi < 2):
        if tentativi == 0:
            user_input = input("\nCosa desideri prenotare oggi? ")
        else:
            user_input = input("\nNon ho capito completamente. Dimmi quale specialità medica e/o quale dottore stai cercando: ")
        
        # Analizza la richiesta dell'utente
        specialita_trovata, dottore_trovato = analizza_richiesta(user_input)
        
        # Aggiorna se abbiamo trovato qualcosa
        if specialita_trovata and not specialita:
            specialita = specialita_trovata
            print(f"Ho capito, stai cercando la specialità: {specialita}")
        
        if dottore_trovato and not dottore:
            dottore = dottore_trovato
            print(f"Ho capito, stai cercando il dottore: {dottore}")
        
        # Se ancora non abbiamo la specialità dopo un tentativo, chiedi esplicitamente
        if not specialita and tentativi > 0:
            specialita_diretta = input("Quale specialità medica ti serve? ")
            if specialita_diretta.strip():
                specialita = specialita_diretta
        
        # Se manca ancora il dottore dopo un tentativo, chiedi se ha preferenze
        if specialita and not dottore and tentativi > 0:
            preferenza = input("Hai una preferenza per un dottore specifico? (sì/no): ")
            if preferenza.lower() in ["no", "n"]:
                print("Va bene, ti mostrerò tutte le opzioni disponibili per questa specialità.")
                break  # Esci dal ciclo anche senza dottore
            elif preferenza.lower() in ["sì", "si", "s", "y", "yes"]:
                dottore_diretta = input("Quale dottore preferisci? ")
                if dottore_diretta.strip():
                    dottore = dottore_diretta
        
        tentativi += 1
    
    # Crea la query di ricerca
    query_specialita = specialita if specialita else ""
    query_dottore = dottore if dottore else ""
    
    # Esegui la query per trovare i 3 risultati più simili
    print(f"\nRicerca in corso per specialità: '{query_specialita}' e dottore: '{query_dottore}'")
    
    # Crea embedding per la ricerca
    if query_dottore and query_specialita:
        query_embedding = get_openai_embeddings(f"{query_specialita} {query_dottore}")
    elif query_specialita:
        query_embedding = get_openai_embeddings(query_specialita)
    else:
        query_embedding = get_openai_embeddings(query_dottore)
    
    # Esegui la query
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    # Mostra i risultati
    print("\n===== Risultati della ricerca =====")
    for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"{i+1}. {doc}")
    
    return results

# Avvia la chat interattiva
if __name__ == "__main__":
    # Carica i dati dal CSV
    csv_path = "availabilities/data.csv"
    doctor_data = load_data_from_csv(csv_path)
    print(f"Caricati {len(doctor_data)} record di dottori e specialità")

    # Genera embeddings per ogni record
    for i, record in enumerate(doctor_data):
        record["id"] = f"doc_{i}"
        record["specialita_embeddings"] = get_openai_embeddings(record["specialita"])
        record["dottore_embeddings"] = get_openai_embeddings(record["dottore"])
        
        # Inserisci nel database Chroma
        collection.upsert(
            ids=[record["id"]],
            documents=[f"Specialità: {record['specialita']}, Dottore: {record['dottore']}"],
            embeddings=[record["specialita_embeddings"]],  # Puoi scegliere quale embedding usare qui
            metadatas=[{
                "specialita": record["specialita"],
                "dottore": record["dottore"]
            }]
        )
    
    # Avvia la chat interattiva
    interactive_chat()