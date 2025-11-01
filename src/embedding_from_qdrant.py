import os
import json
from datetime import datetime, timezone
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text as sql_text
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams

# -------------------- Настройка --------------------
load_dotenv()

PG_CONN = os.getenv("PG_CONN")
assert PG_CONN, "PG_CONN не задан в окружении"

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # может быть None
COLLECTION_NAME = "final_db"
BATCH_SIZE = 500

VECTOR_SIZE = 384
EMBEDDER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# -------------------- Подключения --------------------
engine = create_engine(PG_CONN)
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)
model = SentenceTransformer(EMBEDDER_NAME)

# -------------------- Коллекция (named vectors) --------------------
existing = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME not in existing:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "text": VectorParams(size=VECTOR_SIZE, distance="Cosine"),
            "meta": VectorParams(size=VECTOR_SIZE, distance="Cosine"),
        },
    )
    print(f"✅ Создана коллекция с named vectors: {COLLECTION_NAME}")
else:
    print(f"ℹ️  Коллекция уже существует: {COLLECTION_NAME}")

# -------------------- Вспомогательные --------------------
def to_epoch_ts(dt) -> int | None:
    if pd.isna(dt) or dt is None:
        return None
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except Exception:
            return None
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    return None


def build_meta_text(row) -> str:
    ch  = row.get("channel_id")
    dt  = row.get("date_utc")
    lang = row.get("lang") or ""
    sent = row.get("sentiment")
    kw   = row.get("k_words")
    react = row.get("social_reactions")  # уже числовое значение -1..1

    return (
        f"channel_id: {ch} | lang: {lang} | date_utc: {dt} | "
        f"reaction_score: {react} | sentiment: {sent} | keywords: {kw}"
    )


# -------------------- Основной пайплайн --------------------
def main():
    query = """
        SELECT
            message_id,
            text_clean,
            date_utc,
            channel_id,
            social_reactions,
            sentiment,
            confidence,
            k_words
        FROM final_db
        WHERE processed = FALSE
        ORDER BY date_utc NULLS LAST, message_id
    """
    df = pd.read_sql(query, engine)

    if df.empty:
        print("Нет новых сообщений для обработки.")
        return

    print(f"Найдено {len(df)} новых сообщений. Индексация в Qdrant…")

    for start in range(0, len(df), BATCH_SIZE):
        batch_df = df.iloc[start:start + BATCH_SIZE].copy()

        # тексты
        texts = batch_df["text_clean"].fillna("").astype(str).tolist()
        # meta-тексты
        meta_texts = batch_df.apply(build_meta_text, axis=1).tolist()

        # эмбеддинги
        emb_text = model.encode(texts, show_progress_bar=False)
        emb_meta = model.encode(meta_texts, show_progress_bar=False)

        points = []
        for i, (_, row) in enumerate(batch_df.iterrows()):
            msg_id = int(row["message_id"])
            ch_id  = int(row["channel_id"]) if pd.notna(row["channel_id"]) else None
            dt     = row.get("date_utc")
            ts     = to_epoch_ts(dt)
            lang = row.get("lang") or "ru"

            # sentiment (строки → числа)
            sent_label = None
            sent_value = (row.get("sentiment") or "").strip().lower()
            if sent_value in ("negative", "positive", "neutral"):
                mapping = {"negative": -1, "neutral": 0, "positive": 1}
                sent_label = mapping[sent_value]

            sent_conf  = float(row["confidence"]) if pd.notna(row.get("confidence")) else None

            payload = {
                "message_id": msg_id,
                "text": row["text_clean"],
                "date": str(dt) if dt is not None else None,
                "ts": ts,
                "channel_id": ch_id,
                "lang": lang,
                "sentiment": sent_label,
                "sentiment_label": sent_value,
                "confidence": sent_conf,
                "reaction_score": float(row["social_reactions"]) if pd.notna(row.get("social_reactions")) else None,
            }

            points.append(
                PointStruct(
                    id=msg_id,
                    vector={"text": emb_text[i].tolist(), "meta": emb_meta[i].tolist()},
                    payload=payload,
                )
            )

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  • Батч {start}-{start + len(points)}: загружено {len(points)} точек")

    # Отмечаем обработанными
    message_ids = df["message_id"].tolist()
    if message_ids:
        update_query = sql_text("""
            UPDATE final_db
            SET processed = TRUE
            WHERE message_id = ANY(:ids)
        """)
        with engine.begin() as conn:
            conn.execute(update_query, {"ids": message_ids})
        print(f"✅ Отмечено как обработано {len(message_ids)} сообщений.")

if __name__ == "__main__":
    main()
