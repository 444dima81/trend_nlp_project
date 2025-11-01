import os
import json
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
PG_CONN = os.getenv("PG_CONN")
engine = create_engine(PG_CONN)

# --- классификация эмодзи ---
positive_emojis = {
    "❤️","❤️‍🔥","😍","🥰","💋","👍","🔥","👏","😁","🎉",
    "🤩","🙏","👌","💯","🤣","🏆","😘","🤗","🤓","👻","🆒","🍓","🍾"
}

negative_emojis = {
    "😢","👎","🤬","🤮","💩","🥱","🥴","💔","🖕","😈","😭",
    "😡","🤡"
}

neutral_emojis = {
    "🤔","🤯","😱","🤨","😐","👀","🙈","🌚","🐳","🌭","⚡️","🗿",
    "🤷‍♂️","🤷","🤷‍♀️","👨‍💻","😴","🎃"
}


def compute_social_reaction(reactions):
    """Подсчёт social_reactions [-1, 1]"""
    # Если пусто, None, NaN, пустая строка — возвращаем None
    if not reactions or reactions in ("null", "None", "", []):
        return None

    # Попытка распарсить JSON
    if isinstance(reactions, str):
        try:
            reactions = json.loads(reactions)
        except json.JSONDecodeError:
            return None

    # Если после парсинга не список — пропускаем
    if not isinstance(reactions, list):
        return None

    pos, neg = 0, 0
    for r in reactions:
        # иногда Telegram API может отдать неожиданные объекты
        if not isinstance(r, dict):
            continue
        emoji = r.get("emoji")
        count = r.get("count", 0)
        if emoji in positive_emojis:
            pos += count
        elif emoji in negative_emojis:
            neg += count

    total = pos + neg
    if total == 0:
        return None

    return round((pos - neg) / total, 3)


def main():
    # Проверяем, есть ли колонка
    with engine.begin() as conn:
        conn.execute(text("""
            ALTER TABLE clean_posts
            ADD COLUMN IF NOT EXISTS social_reactions FLOAT;
        """))

    # Загружаем реакции
    df = pd.read_sql("SELECT message_id, reactions FROM messages", engine)
    print(f"Найдено {len(df)} записей для обработки.")

    # Считаем social_reactions
    df["social_reactions"] = df["reactions"].apply(compute_social_reaction)

    # Обновляем в БД
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(
                text("UPDATE clean_posts SET social_reactions = :sr WHERE message_id = :mid"),
                {"sr": row["social_reactions"], "mid": row["message_id"]}
            )

    print("✅ social_reactions успешно добавлен в базу.")


if __name__ == "__main__":
    main()



#     -- 1. Общая статистика
# SELECT
#     COUNT(*) AS total_posts,
#     COUNT(social_reactions) AS posts_with_reactions,
#     ROUND(AVG(social_reactions)::numeric, 3) AS avg_social_score,
#     ROUND(MIN(social_reactions)::numeric, 3) AS min_social_score,
#     ROUND(MAX(social_reactions)::numeric, 3) AS max_social_score,
#     SUM(CASE WHEN social_reactions > 0 THEN 1 ELSE 0 END) AS positive_posts,
#     SUM(CASE WHEN social_reactions < 0 THEN 1 ELSE 0 END) AS negative_posts,
#     SUM(CASE WHEN social_reactions = 0 THEN 1 ELSE 0 END) AS neutral_posts
# FROM clean_posts;

# -- 2. Статистика по каналам
# SELECT
#     channel_id,
#     COUNT(*) AS total_posts,
#     COUNT(social_reactions) AS posts_with_reactions,
#     ROUND(AVG(social_reactions)::numeric, 3) AS avg_social_score,
#     SUM(CASE WHEN social_reactions > 0 THEN 1 ELSE 0 END) AS positive_posts,
#     SUM(CASE WHEN social_reactions < 0 THEN 1 ELSE 0 END) AS negative_posts,
#     SUM(CASE WHEN social_reactions = 0 THEN 1 ELSE 0 END) AS neutral_posts
# FROM clean_posts
# GROUP BY channel_id
# ORDER BY avg_social_score DESC;