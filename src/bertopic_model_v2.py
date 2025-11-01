import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from qdrant_client import QdrantClient
from bertopic import BERTopic

# -------------------- Настройка --------------------
load_dotenv()
PG_CONN = os.getenv("PG_CONN")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "telegram_posts_v2"

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
    texts.append(payload.get("text", ""))
    message_ids.append(point.id)

# Извлекаем чистые числовые вектора
clean_vectors = []
for v in vectors:
    if isinstance(v, dict):
        # Если вектор хранится как {'embedding': [...]}
        # или {'vector': [...]}, попробуй оба варианта
        v = v.get("embedding") or v.get("vector") or list(v.values())[0]
    clean_vectors.append(v)

vectors = np.array(clean_vectors, dtype=np.float32)
print(f"Загружено {len(vectors)} точек для обучения BERTopic")

# -------------------- Обучение BERTopic --------------------
print("🧠 Обучаем BERTopic...")
topic_model = BERTopic(language="multilingual", calculate_probabilities=True, verbose=True)
topics, probs = topic_model.fit_transform(texts, vectors)

# -------------------- DataFrame с результатами --------------------
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

# Чистим null
df_merged['social_reactions'] = df_merged['social_reactions'].fillna(0)
df_merged['confidence'] = df_merged['confidence'].fillna(0)
df_merged['views'] = df_merged['views'].fillna(0).astype(int)
df_merged['sentiment'] = df_merged['sentiment'].fillna('neutral')
df_merged['text_clean'] = df_merged['text_clean'].fillna("")

# -------------------- Recency weight --------------------
df_merged['date_utc'] = pd.to_datetime(df_merged['date_utc'])
df_merged['days_ago'] = (pd.Timestamp.now(tz='UTC') - df_merged['date_utc']).dt.days
# Экспоненциальное затухание веса: посты старше 14 дней почти не учитываются
df_merged['recency_weight'] = np.exp(-df_merged['days_ago'] / 7)
df_merged['weighted_views'] = df_merged['views'] * df_merged['recency_weight']
df_merged['weighted_reactions'] = df_merged['social_reactions'] * df_merged['recency_weight']

# -------------------- Преобразуем sentiment в числовое значение --------------------
sentiment_map = {'negative': -1, 'neutral': 0, 'positive': 1}
df_merged['sentiment_score'] = df_merged['sentiment'].map(sentiment_map)
df_merged['weighted_sentiment'] = df_merged['sentiment_score'] * df_merged['recency_weight']

# -------------------- Агрегация по теме с учётом recency_weight --------------------
topic_summary = (
    df_merged.groupby("topic")
    .agg(
        weighted_views=("weighted_views", "sum"),
        mean_weighted_sentiment=("weighted_sentiment", "mean"),
        mean_weighted_reaction=("weighted_reactions", "mean"),
        mean_confidence=("confidence", "mean"),
        n_posts=("message_id", "count")
    )
    .reset_index()
    .sort_values("weighted_views", ascending=False)
)

# Добавляем пример поста для каждой темы
examples = (
    df_merged.groupby("topic")
    .apply(lambda x: x['text_clean'].sample(1).values[0] if not x['text_clean'].empty else "")
    .reset_index(name='example_post')
)
topic_summary = topic_summary.merge(examples, on="topic", how="left")
# -------------------- Сохраняем в CSV --------------------
topic_summary.to_csv("../data/bertopic_model_v2.csv", index=False)
print("✅ Отчёт сохранён в bertopic_model_v2.csv")

MODEL_PATH = "../models/bertopic_model_v2"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
topic_model.save(MODEL_PATH)
print(f"✅ Модель BERTopic сохранена в {MODEL_PATH}")

# Подключается к Qdrant (telegram_posts_v2),
#  • Берёт вектора и тексты,
#  • Обучает BERTopic на готовых эмбеддингах (ускорение ×3–5),
#  • Объединяет результаты с таблицей clean_posts,
#  • Добавляет метрики (views, social_reactions, sentiment).