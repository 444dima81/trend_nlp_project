import re
import pandas as pd
import emoji
import os
import json
from langdetect import detect
from sqlalchemy import create_engine, text
import sqlalchemy.types as sqltypes
from dotenv import load_dotenv
import logging

# ===================== Логирование =====================
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ===================== Конфиг =====================
load_dotenv()
PG_CONN = os.getenv("PG_CONN")

# --- Все стоп-паттерны ---
STOP_PATTERNS = [
    # старые кастомные
    r"[\s\W🐚🔹📢🤑👍]*бэкдор",
    r"[\s\W🐚🔹📢🤑👍]*подписаться на риа новости",
    r"[\s\W🐚🔹📢🤑👍]*читать рбк в telegram",
    r"[\s\W🐚🔹📢🤑👍]*больше инфографики.*?рбк",
    r"[\s\W🐚🔹📢🤑👍]*картина дня.*?рбк",
    r"[\s\W🐚🔹📢🤑👍]*другие видео этого дня.*?рбк",
    r"[\s\W🐚🔹📢🤑👍]*следите за новостями.*?рбк",
    r"[\s\W🐚🔹📢🤑👍]*прямой эфир",
    r"[\s\W🐚🔹📢🤑👍]*the экономист",

    # новые и расширенные
    r"ИИ by AIvengo",
    r"^Отправ(ь|ить) .*",           # если начинается с "Отправь"
    r"читайте в .*",                # обрезаем все упоминания "Читайте в ..."
    r"— в материале РБК",
    r"Фото: .*",
    r"— читайте в подписке РБК",
    r"Самые важные новости — в телеграм-канале РБК",
    r"– в инфографике РИА Новости"
]

# --- Полное удаление новостей ---
FULL_REMOVE_PATTERNS = [
    r"Читайте самые интересные",
    r"Главные новости",
    r"Утренний выпуск новостей",
    r"большой розыгрыш"
]

# --- Регулярка для рекламы ---
REKLAMA_PATTERN = re.compile(
    r"(?i)(?:реклама[\s\.\-:]*ооо|ооо[\s\.\-:]*реклама)"
)


# ===================== Очистка текста =====================

def remove_urls(text):
    return re.sub(r'http\S+|www\S+', '', text)

def remove_mentions(text):
    return re.sub(r'@\w+', '', text)

def remove_emojis(text: str) -> str:
    return emoji.replace_emoji(text, replace='')

def remove_html_tags(text):
    return re.sub(r'<.*?>', '', text)

def remove_stop_phrases(text: str) -> str:
    """Удаляет или обрезает по стоп-паттернам"""
    for pattern in STOP_PATTERNS:
        text = re.split(pattern, text, flags=re.IGNORECASE)[0]
    return text.strip()

def remove_full_news(text: str) -> bool:
    """Полностью исключает новости, если они содержат стоп-фразу"""
    if not text:
        return True
    for pattern in FULL_REMOVE_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False

def clean_text(text):
    if not text:
        return ""
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_emojis(text)
    text = remove_html_tags(text)
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = remove_stop_phrases(text)
    return text

def detect_language(text):
    try:
        return detect(text) if len(text.split()) > 2 else None
    except:
        return None


# ===================== Обработка DataFrame =====================

def process_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Удаляем рекламу
    df = df[~df['text'].str.contains(REKLAMA_PATTERN, na=False)]

    # Удаляем новости по полным паттернам
    df = df[~df['text'].apply(remove_full_news)]

    # Чистим текст
    df['text_clean'] = df['text'].apply(clean_text)

    # Удаляем короткие сообщения (<3 слов)
    df = df[df['text_clean'].str.split().str.len() >= 3]

    # Определяем язык
    df['lang'] = df['text_clean'].apply(detect_language)

    # Преобразуем реакции в JSON
    df['reactions'] = df['reactions'].apply(
        lambda d: json.dumps(d, ensure_ascii=False) if isinstance(d, dict) else None
    )

    # Удаляем дубликаты по очищенному тексту
    df = df.drop_duplicates(subset=['text_clean'])
    return df


# ===================== Основная функция =====================

def main():
    engine = create_engine(PG_CONN)
    query = """
        SELECT channel_id, message_id, date_utc, text, views, reactions
        FROM messages
        WHERE processed = FALSE;
    """
    df = pd.read_sql(query, engine)

    if df.empty:
        log.info("Нет новых сообщений для обработки.")
        return

    df_clean = process_df(df)

    dtype_dict = {
        'channel_id': sqltypes.BigInteger(),
        'message_id': sqltypes.BigInteger(),
        'date_utc': sqltypes.DateTime(),
        'text_clean': sqltypes.Text(),
        'lang': sqltypes.String(length=10),
        'views': sqltypes.BigInteger(),
        'reactions': sqltypes.JSON
    }

    df_clean.to_sql('final_db', engine, if_exists='append', index=False, dtype=dtype_dict)
    log.info(f"Обработано {len(df_clean)} записей.")


if __name__ == "__main__":
    main()