import os
import asyncio
import json
from datetime import datetime, timezone

from dotenv import load_dotenv
from telethon import TelegramClient
from langdetect import detect, DetectorFactory, LangDetectException
import asyncpg

# Детерминированность langdetect
DetectorFactory.seed = 0

load_dotenv()
API_ID = int(os.getenv("TG_API_ID"))
API_HASH = os.getenv("TG_API_HASH")
SESSION = os.getenv("TG_SESSION_FILE", "/home/nlp_project/data/parser.session")

client = TelegramClient(SESSION, API_ID, API_HASH)

PG_DSN = os.getenv("PG_DSN")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "200"))

INSERT_CHANNEL_SQL = """
INSERT INTO channels (channel_id, title, username, raw, last_scraped)
VALUES ($1, $2, $3, $4::jsonb, $5)
ON CONFLICT (channel_id)
DO UPDATE SET title = EXCLUDED.title, username = EXCLUDED.username, raw = EXCLUDED.raw, last_scraped = EXCLUDED.last_scraped;
"""

INSERT_MESSAGE_SQL = """
INSERT INTO messages (channel_id, message_id, text, author_id, author_username, date_utc, views, language,
is_forward, has_media, media_type, reactions, raw)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12::jsonb,$13::jsonb)
ON CONFLICT DO NOTHING;
"""

def detect_language(text):
    if not text or text.strip() == "":
        return None
    try:
        lang = detect(text)
        if lang.startswith("en"):
            return "en"
        elif lang.startswith("ru"):
            return "ru"
        else:
            return "mixed"
    except LangDetectException:
        return None

def safe_json_serialize(obj):
    def json_serializer(obj):
        if isinstance(obj, bytes):
            return clean_text(obj.decode('utf-8', errors='replace'))
        elif isinstance(obj, str):
            return clean_text(obj)
        elif hasattr(obj, 'dict'):
            return obj.dict
        else:
            return clean_text(str(obj))
    return json.dumps(obj, default=json_serializer, ensure_ascii=False)

def clean_text(text):
    if not text:
        return ""
    import re
    cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

async def upsert_channel(conn, entity):
    raw = dict(entity.to_dict())
    title = clean_text(getattr(entity, "title", None) or "")
    username = clean_text(getattr(entity, "username", None) or "")
    await conn.execute(
        INSERT_CHANNEL_SQL,
        entity.id,
        title,
        username,
        safe_json_serialize(raw),
        datetime.now(timezone.utc)
    )

async def fetch_and_store(channel_identifier, limit_per_run=10000, until_date=None):
    await client.start()
    entity = await client.get_entity(channel_identifier)
    channel_id = entity.id

    pg = await asyncpg.create_pool(dsn=PG_DSN, min_size=1, max_size=4)

    async with pg.acquire() as conn:
        await upsert_channel(conn, entity)
        row = await conn.fetchrow(
            "SELECT max(message_id) AS max_id FROM messages WHERE channel_id = $1;",
            channel_id
        )
        last_id = row["max_id"] or 0

    to_insert = []
    cnt = 0
    async for msg in client.iter_messages(entity, reverse=False):
        if msg.id <= last_id:
            continue

        # Остановка по дате
        if until_date and msg.date <= until_date:
            print(f"Reached target date {until_date}. Stopping parsing.")
            break

        author = getattr(msg, "sender", None)

        # Сбор реакций
        reactions_data = None
        if msg.reactions:
            reactions_data = []
            for r in msg.reactions.results:
                if hasattr(r.reaction, 'emoticon'):
                    reactions_data.append({"emoji": r.reaction.emoticon, "count": r.count})
                elif hasattr(r.reaction, 'emoticon_id'):
                    reactions_data.append({"emoji": str(r.reaction.emoticon_id), "count": r.count})
                else:
                    reactions_data.append({"emoji": str(r.reaction), "count": r.count})

        views = getattr(msg, "views", None)
        language = detect_language(msg.message)

        rec = (
            channel_id,
            msg.id,
            clean_text(msg.message or ""),
            author.id if author else None,
            clean_text(getattr(author, "username", None) or ""),
            msg.date.replace(tzinfo=timezone.utc),
            views,
            language,
            bool(msg.fwd_from),
            bool(msg.media),
            type(msg.media).__name__ if msg.media else None,
            safe_json_serialize(reactions_data),
            safe_json_serialize(msg.to_dict())
        )
        to_insert.append(rec)
        cnt += 1

        if len(to_insert) >= BATCH_SIZE:
            async with pg.acquire() as conn:
                async with conn.transaction():
                    await conn.executemany(INSERT_MESSAGE_SQL, to_insert)
            to_insert.clear()

        if cnt >= limit_per_run:
            print(f"Reached limit {limit_per_run}. Stopping parsing.")
            break

    if to_insert:
        async with pg.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(INSERT_MESSAGE_SQL, to_insert)

    async with pg.acquire() as conn:
        await conn.execute(
            "UPDATE channels SET last_scraped = $1 WHERE channel_id = $2;",
            datetime.now(timezone.utc),
            channel_id
        )

    await pg.close()
    print(f"Inserted/queued {cnt} new messages for channel {channel_identifier} (id={channel_id}).")

async def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python tg_to_pg.py <channel1> [<channel2> ...]")
        return

    channels = sys.argv[1:]
    await client.start()

    # Дата два года назад от 23/10/2025
    until_date = datetime(2023, 10, 23, tzinfo=timezone.utc)
    limit_per_run = 10000

    for ch in channels:
        try:
            await fetch_and_store(
                channel_identifier=ch,
                limit_per_run=limit_per_run,
                until_date=until_date
            )
        except Exception as e:
            print(f"Error for {ch}: {e}")

if __name__ == "__main__":
    asyncio.run(main())