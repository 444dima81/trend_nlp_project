import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct

load_dotenv()

# -------------------- Настройки --------------------
PG_CONN = os.getenv("PG_CONN")
assert PG_CONN, "PG_CONN не задан в окружении"

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # может быть None

TOPICS_COLLECTION = "topics_summary"
VECTOR_SIZE = 384
EMBEDDER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 200

# -------------------- Подключения --------------------
engine = create_engine(PG_CONN)
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)
model = SentenceTransformer(EMBEDDER_NAME)

# -------------------- Коллекция для топиков --------------------
existing = [c.name for c in client.get_collections().collections]
if TOPICS_COLLECTION not in existing:
    client.create_collection(
        collection_name=TOPICS_COLLECTION,
        vectors_config={
            "text": VectorParams(size=VECTOR_SIZE, distance="Cosine"),
        },
    )
    print(f"✅ Создана коллекция для топиков: {TOPICS_COLLECTION}")
else:
    print(f"ℹ️  Коллекция уже существует: {TOPICS_COLLECTION}")

# -------------------- Загружаем CSV с топиками --------------------
df_topics = pd.read_csv("../data/topic_df.csv")
df_topics['text_for_embedding'] = df_topics['subject'] + ". " + df_topics['example_post']

# -------------------- Генерация эмбеддингов --------------------
vectors = model.encode(df_topics['text_for_embedding'].tolist(), convert_to_numpy=True, show_progress_bar=True)

# -------------------- Формируем точки --------------------
points = [
    PointStruct(
        id=int(row['topic']) if int(row['topic']) >= 0 else abs(int(row['topic'])) + 1000000,  # чтобы избежать отрицательных ID
        vector={"text": vec.tolist()},  
        payload={
            "subject": row['subject'],
            "mean_reaction": row['mean_reaction'],
            "mean_sentiment": row['mean_sentiment'],
            "mean_confidence": row['mean_confidence'],
            "avg_views": row['avg_views'],
            "n_posts": row['n_posts'],
            "example_post": row['example_post']
        }
    )
    for row, vec in zip(df_topics.to_dict(orient='records'), vectors)
]

# -------------------- Загружаем в Qdrant --------------------
for i in range(0, len(points), BATCH_SIZE):
    batch = points[i:i+BATCH_SIZE]
    client.upsert(collection_name=TOPICS_COLLECTION, points=batch)

print(f"✅ Загружено {len(points)} топиков в коллекцию {TOPICS_COLLECTION}")