import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import logging

# -------------------- Настройка --------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

load_dotenv()
PG_CONN = os.getenv("PG_CONN")

# Папка для сохранения Parquet
PARQUET_DIR = "clean_posts_parquet/"

# -------------------- Функции --------------------
def fetch_clean_posts(engine):
    """
    Подтягиваем все очищенные посты из clean_posts
    """
    query = "SELECT * FROM clean_posts"
    df = pd.read_sql(query, engine)
    return df

def save_to_parquet(df, path=PARQUET_DIR, partition_by_date=True):
    """
    Сохраняем DataFrame в Parquet.
    Если partition_by_date=True, создаём папки по дате.
    """
    if df.empty:
        log.info("Нет данных для сохранения.")
        return
    
    if partition_by_date and 'date_utc' in df.columns:
        # приводим к дате без времени
        df['date_only'] = pd.to_datetime(df['date_utc']).dt.date
        df.to_parquet(
            path,
            engine="pyarrow",
            index=False,
            compression="snappy",
            partition_cols=["date_only"]
        )
        log.info(f"Сохранено {len(df)} записей с разбиением по дате в {path}")
    else:
        df.to_parquet(
            os.path.join(path, "clean_posts.parquet"),
            engine="pyarrow",
            index=False,
            compression="snappy"
        )
        log.info(f"Сохранено {len(df)} записей в {path}")

# -------------------- Основной скрипт --------------------
def main():
    engine = create_engine(PG_CONN)
    
    log.info("Подтягиваем данные из clean_posts...")
    df_clean = fetch_clean_posts(engine)
    
    log.info(f"Получено {len(df_clean)} записей.")
    
    save_to_parquet(df_clean)

if __name__ == "__main__":
    main()