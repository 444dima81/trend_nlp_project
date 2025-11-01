import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from qdrant_client import QdrantClient
from bertopic import BERTopic
import matplotlib.pyplot as plt

# -------------------- Настройка --------------------
load_dotenv()
PG_CONN = os.getenv("PG_CONN")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "telegram_posts_v2"
MODEL_PATH = "../models/bertopic_model_v3"  # путь для сохранения модели

# -------------------- Подключение --------------------
engine = create_engine(PG_CONN)
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# -------------------- Загрузка данных из Qdrant --------------------
print("📥 Загружаем данные из Qdrant...")
points = client.scroll(
    collection_name=COLLECTION_NAME,
    with_vectors=True,
    with_payload=True,
    limit=60000
)

vectors = []
texts = []
message_ids = []

for point in points[0]:
    vectors.append(point.vector)
    payload = point.payload or {}
    # используем очищенный текст
    texts.append(payload.get("text_clean", "") or payload.get("text", ""))
    message_ids.append(point.id)

# Извлекаем чистые числовые вектора
clean_vectors = []
for v in vectors:
    if isinstance(v, dict):
        v = v.get("embedding") or v.get("vector") or list(v.values())[0]
    clean_vectors.append(v)

vectors = np.array(clean_vectors, dtype=np.float32)
print(f"Загружено {len(vectors)} точек для обучения BERTopic")

# -------------------- Обучение BERTopic --------------------
print("🧠 Обучаем BERTopic...")
topic_model = BERTopic(language="multilingual", calculate_probabilities=True, verbose=True)
topics, probs = topic_model.fit_transform(texts, vectors)

# -------------------- Сохраняем модель --------------------
topic_model.save(MODEL_PATH)
print(f"✅ Модель BERTopic сохранена в {MODEL_PATH}")

# -------------------- Создаём DataFrame с результатами --------------------
df_topics = pd.DataFrame({
    "message_id": message_ids,
    "topic": topics,
    "probability": [p.max() if p is not None else 0 for p in probs],
})

# -------------------- Подгружаем метаданные из clean_posts --------------------
query = """
SELECT
    message_id,
    text_clean,
    date_utc,
    views,
    channel_id,
    social_reactions,
    sentiment,
    confidence,
    k_words
FROM clean_posts;
"""
df_clean = pd.read_sql(query, engine)

# -------------------- Объединяем --------------------
df_merged = pd.merge(df_topics, df_clean, on="message_id", how="left")
df_merged.fillna({"social_reactions": 0, "confidence": 0, "views": 0, "sentiment": "neutral", "text_clean": ""}, inplace=True)
sentiment_map = {'negative': -1, 'neutral': 0, 'positive': 1}
df_merged['sentiment_score'] = df_merged['sentiment'].map(sentiment_map)

# -------------------- Агрегация по теме --------------------
topic_summary = (
    df_merged.groupby("topic")
    .agg(
        mean_reaction=("social_reactions", "mean"),
        mean_sentiment=("sentiment_score", "mean"),
        mean_confidence=("confidence", "mean"),
        avg_views=("views", "mean"),
        n_posts=("message_id", "count")
    )
    .reset_index()
    .sort_values("n_posts", ascending=False)
)

# Добавляем пример поста для каждой темы
examples = (
    df_merged.groupby("topic")
    .apply(lambda x: x['text_clean'].dropna().sample(1).values[0] if not x['text_clean'].dropna().empty else "")
    .reset_index(name='example_post')
)
topic_summary = topic_summary.merge(examples, on="topic", how="left")

# -------------------- Вывод топ-15 тем --------------------
top15 = topic_summary.head(15)
print("🔥 Топ-15 тем:")
print(top15[["topic", "n_posts", "mean_sentiment", "mean_reaction", "example_post"]])

# -------------------- Сохраняем CSV --------------------
topic_summary.to_csv("../data/bertopic_model_v3.csv", index=False)
print("✅ Отчёт сохранён в bertopic_model_v3.csv")

# -------------------- Визуализация UMAP --------------------
umap_emb = topic_model.visualize_topics()
umap_emb.show()

#  • Загружает до 60 000 точек из Qdrant (вектор + payload),
#  • Использует text_clean и embedding напрямую,
#  • Обучает BERTopic (с внешними векторами),
#  • Объединяет с clean_posts,
#  • Вычисляет:
#  • mean_reaction — средние реакции,
#  • mean_sentiment — среднее настроение,
#  • mean_confidence — уверенность в сентименте,
#  • avg_views — средние просмотры,
#  • n_posts — кол-во постов в теме,
#  • Добавляет пример поста и сохраняет в CSV.