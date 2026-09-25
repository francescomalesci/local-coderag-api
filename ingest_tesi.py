import chromadb
from chromadb.utils import embedding_functions

# 1. Configurazione
CHROMA_DATA_PATH = "./chroma_db"
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Creiamo una NUOVA collezione per non mescolare i dati
collection = client.get_or_create_collection(
    name="dati_tesi",
    embedding_function=embedding_model
)

# 2. I tuoi appunti reali
testo_tesi = """Durante lo sviluppo della tesi sulla segmentazione dei binari ferroviari con YOLO e SAM 2, ho riscontrato un problema con i sassi e la ghiaia ai bordi della massicciata, che venivano spesso classificati come binario (falsi positivi). 
Per risolvere, ho introdotto un meccanismo di pesi spaziali (spatial weighting). 
Ho applicato un peso maggiore ai pixel centrali del binario reale. In questo modo, durante il training, la rete neurale viene penalizzata pesantemente se sbaglia la previsione al centro della rotaia, mentre gli errori sui bordi esterni (la ghiaia) vengono ignorati o pesano pochissimo sulla loss function."""

# 3. Chunking e Inserimento
# Essendo un testo breve, lo inseriamo come blocco unico
collection.add(
    documents=[testo_tesi],
    metadatas=[{"source": "appunti_tesi.txt", "argomento": "falsi positivi"}],
    ids=["tesi_chunk_1"]
)

print("Appunti della tesi elaborati e salvati nel database vettoriale.")