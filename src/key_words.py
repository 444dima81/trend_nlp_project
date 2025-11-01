import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from stop_words import get_stop_words
from tqdm import tqdm

# -------------------- Настройка --------------------
load_dotenv()

PG_CONN = os.getenv("PG_CONN")
BATCH_SIZE = 500  # чуть больше — ускорит общий прогон

# -------------------- Подключение к БД --------------------
engine = create_engine(PG_CONN)

# -------------------- Инициализация моделей --------------------
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
kw_model = KeyBERT(model=embed_model)

# -------------------- Проверка и создание столбца --------------------
with engine.begin() as conn:
    conn.execute(text("""
        ALTER TABLE clean_posts
        ADD COLUMN IF NOT EXISTS k_words TEXT
    """))

# -------------------- Основные функции --------------------
stopwords_ru = get_stop_words("russian")

def extract_keywords(text, top_n=5, ngram_range=(1, 2)):
    """Извлекает ключевые слова для текста"""
    if not text or pd.isna(text) or len(text.split()) < 3:
        return None
    try:
        kws = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=ngram_range,
            stop_words=stopwords_ru,
            top_n=top_n
        )
        if not kws:
            return None
        return ", ".join([kw for kw, _ in kws])
    except Exception as e:
        print(f"⚠️ Ошибка при обработке текста: {e}")
        return None


def main():
    total_processed = 0
    offset = 0

    while True:
        query = text("""
            SELECT message_id, text_clean
            FROM clean_posts
            WHERE k_words IS NULL
            ORDER BY message_id
            LIMIT :limit OFFSET :offset
        """)
        df = pd.read_sql(query, engine, params={"limit": BATCH_SIZE, "offset": offset})
        if df.empty:
            print(f"✅ Все посты обработаны. Всего обработано: {total_processed}")
            break

        print(f"⚙️ Обрабатываем батч {offset + 1} — {offset + len(df)}")

        # Генерация ключевых слов
        df["k_words"] = [extract_keywords(t) for t in tqdm(df["text_clean"], desc="Извлекаем ключевые слова")]

        # Массовое обновление (вместо поштучного)
        with engine.begin() as conn:
            temp_df = df.dropna(subset=["k_words"])
            if not temp_df.empty:
                values = [
                    {"message_id": mid, "k_words": kw}
                    for mid, kw in zip(temp_df["message_id"], temp_df["k_words"])
                ]
                conn.execute(
                    text("""
                        UPDATE clean_posts
                        SET k_words = :k_words
                        WHERE message_id = :message_id
                    """),
                    values,
                )

        total_processed += len(df)
        offset += BATCH_SIZE


if __name__ == "__main__":
    main()