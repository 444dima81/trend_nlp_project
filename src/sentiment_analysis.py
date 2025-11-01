import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.special import softmax

# -------------------- Настройка --------------------
load_dotenv()
PG_CONN = os.getenv("PG_CONN")
BATCH_SIZE = 256  # размер батча для предсказаний

# -------------------- Подключение к БД --------------------
engine = create_engine(PG_CONN)

# -------------------- Создание колонок, если их нет --------------------
with engine.begin() as conn:
    conn.execute(text("""
        ALTER TABLE final_db
        ADD COLUMN IF NOT EXISTS sentiment TEXT;
    """))
    conn.execute(text("""
        ALTER TABLE final_db
        ADD COLUMN IF NOT EXISTS confidence FLOAT;
    """))

# -------------------- Загрузка модели --------------------
MODEL_NAME = "blanchefort/rubert-base-cased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()  # отключаем градиенты

labels = ["negative", "neutral", "positive"]

# -------------------- Функция предсказания --------------------
def get_sentiment_batch(texts):
    """Возвращает список (sentiment, confidence) для батча текстов"""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    with torch.no_grad():
        outputs = model(**inputs)
        scores = softmax(outputs.logits.numpy(), axis=1)
        sentiments = [labels[s.argmax()] for s in scores]
        confidences = [float(s.max()) for s in scores]
    return list(zip(sentiments, confidences))

# -------------------- Основной цикл --------------------
def main():
    # Загружаем все посты на русском
    query = "SELECT message_id, text_clean FROM final_db WHERE lang='ru'"
    df = pd.read_sql(query, engine)
    total = len(df)
    print(f"Найдено {total} сообщений для анализа.")

    if df.empty:
        return

    results = []
    for start in range(0, total, BATCH_SIZE):
        batch_df = df.iloc[start:start+BATCH_SIZE]
        batch_texts = batch_df['text_clean'].tolist()
        batch_result = get_sentiment_batch(batch_texts)
        results.extend(batch_result)
        print(f"✅ Обработан батч {start}-{start+len(batch_texts)}")

    # Добавляем результаты в DataFrame
    df['sentiment'], df['confidence'] = zip(*results)

    # Обновляем базу по каждому message_id
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(
                text("""
                    UPDATE final_db
                    SET sentiment = :sentiment,
                        confidence = :confidence
                    WHERE message_id = :mid
                """),
                {"sentiment": row['sentiment'], "confidence": row['confidence'], "mid": int(row['message_id'])}
            )

    # Итоговый отчёт
    summary = df['sentiment'].value_counts()
    print("\n📊 Итоговая статистика тональности:")
    print(summary)
    print(f"\nОбработано {len(df)} сообщений.")

if __name__ == "__main__":
    main()


# 📊 Итоговая статистика тональности:
# sentiment
# negative    44696
# positive    39013
# neutral      5432
# Name: count, dtype: int64

# Обработано 89141 сообщений