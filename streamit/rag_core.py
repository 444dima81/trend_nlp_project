# rag_core.py
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

# ===================== Конфиг =====================
COLLECTIONS = {
    "final_db": ["text", "meta"],
    "topics_summary": ["text"],
}
EMBEDDER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

TOP_K_EACH = 8
FINAL_K = 10
SCORE_THRESHOLD = 0.20
W_TEXT = 0.7
W_META = 0.3
MAX_CONTEXT_CHARS = 6000

# Поле с unix timestamp в payload (можно переопределить через ENV)
DATE_TS_FIELD = os.getenv("DATE_TS_FIELD", "ts")

# Сколько вариантов переписанного запроса учитывать максимум
MAX_QUERY_VARIANTS = 5

PROMPT_SYSTEM = (
    "Ты — эксперт по журналистике и обзору новостей с сильными ораторскими навыками. "
    "Говоришь по-русски чётко и структурированно. Твоя задача: по предоставленному контексту "
    "выявить именно НОВОСТИ и кратко, точно их пересказать.\n\n"
    "Правила:\n"
    "1) Используй ТОЛЬКО факты из контекста; если фактов нет — честно скажи об этом.\n"
    "2) Сначала короткий вывод (1–2 предложения).\n"
    "3) Затем 3–6 буллетов (кто/что/когда/где/почему/что дальше).\n"
    "4) Сопоставляй по словам и по смыслу — у тебя есть оба сигнала.\n"
    # "5) В конце укажи источники по номерам [1], [2], ... из контекста.\n"
)

# ===================== Утилиты =====================
RU_MONTHS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6, "июл": 7,
    "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12
}
SENTIMENT_MAP = {
    "положительн": ("positive", 1), "позитив": ("positive", 1),
    "нейтрал": ("neutral", 0), "отсутствие эмоц": ("neutral", 0),
    "негатив": ("negative", -1), "отрицательн": ("negative", -1),
}
LANG_HINTS = {"русск": "ru", "по-русск": "ru", "english": "en", "английск": "en"}  # на будущее

RE_CHANNEL = re.compile(r"(канал|channel)\s*[:=]?\s*(\d{6,})", re.I)
RE_ID = re.compile(r"\b(channel_id|канал_id|канал)\s*[:=]\s*(\d+)", re.I)
RE_REACT_GE = re.compile(r"(реакц|reaction)[^\d\-]*([\-]?\d+(?:[.,]\d+)?)\s*\+?", re.I)
RE_CONF_GE = re.compile(r"(уверенн|confidence)[^\d\-]*([\-]?\d+(?:[.,]\d+)?)\s*\+?", re.I)
RE_YEAR = re.compile(r"\b(20\d{2})\b")
RE_LAST_N_DAYS = re.compile(r"последн(ие|их)?\s+(\d+)\s*(дн|недел|месяц)", re.I)

# ---- Аббревиатуры/синонимы для переписывания запроса ----
# Ключи — в нижнем регистре; добавляй свои по мере надобности.
ABBR_SYNONYMS: Dict[str, List[str]] = {
    # AI / ИИ
    "ии": ["искусственный интеллект", "ai", "artificial intelligence"],
    "ai": ["искусственный интеллект", "ии", "artificial intelligence"],
    "нейросеть": ["нейросети", "нейронная сеть", "нейронные сети"],
    "нейронка": ["нейросеть", "нейронная сеть"],
    "ml": ["машинное обучение"],
    "машобуч": ["машинное обучение"],

    # Crypto
    "крипта": ["криптовалюта", "crypto", "криптовалюты"],
    "криптов": ["криптовалюта", "криптовалюты", "crypto"],
    "btc": ["биткоин", "bitcoin", "btc"],
    "биток": ["биткоин", "bitcoin", "btc"],
    "eth": ["эфириум", "ethereum", "eth"],

    # Прочие частые
    "смартфон": ["смартфоны", "мобильный телефон"],
    "авто": ["автомобиль", "машина", "автотехника"],
}

def _normalize_ru(text: str) -> str:
    return text.lower().replace("ё", "е")

load_dotenv()
USE_OPENAI = False
openai_client = None
if os.getenv("API_KEY"):
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=os.getenv("API_KEY"), base_url="https://foundation-models.api.cloud.ru/v1")
        USE_OPENAI = True
    except Exception:
        print(f"[GigaChat init error] {e}")
        USE_OPENAI = False

def init_clients() -> Tuple[QdrantClient, SentenceTransformer]:
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    embedder = SentenceTransformer(EMBEDDER_NAME)
    return client, embedder

def encode(embedder: SentenceTransformer, text: str) -> List[float]:
    return embedder.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()

# --------- время из запроса ----------
def parse_time_filter(query: str) -> Optional[datetime]:
    now = datetime.now(timezone.utc)
    patterns = {
        r"за\s+неделю": timedelta(days=7),
        r"за\s+месяц": timedelta(days=30),
        r"за\s+3\s*месяц": timedelta(days=90),
        r"за\s+пол[-\s]*года": timedelta(days=180),
        r"за\s+год": timedelta(days=365),
    }
    for pattern, delta in patterns.items():
        if re.search(pattern, query.lower()):
            return now - delta
    return None

def _to_ts_safe(iso_str: str, end: bool = False) -> Optional[int]:
    """Строковый ISO → unix ts. Лучше иметь готовое int-поле в Qdrant."""
    try:
        dt = datetime.fromisoformat(str(iso_str).replace("Z", "+00:00"))
        if end:
            dt = dt.replace(hour=23, minute=59, second=59)
        return int(dt.replace(tzinfo=timezone.utc).timestamp())
    except Exception:
        return None

# --------- Переписывание запроса (аббревиатуры/синонимы) ----------
def expand_query_variants(query_text: str) -> List[str]:
    """
    Возвращает до MAX_QUERY_VARIANTS вариантов запроса:
    - исходный
    - варианты с заменой найденных аббревиатур
    - варианты с обогащением (добавлением в конец расширений)
    """
    base = query_text.strip()
    t_norm = _normalize_ru(base)

    variants: List[str] = [base]
    seen: set[str] = {t_norm}

    # Сбор всех расширений, которые встречаются в тексте
    enrich_tokens: List[str] = []
    for key, exps in ABBR_SYNONYMS.items():
        if key in t_norm:
            # 1) Подстановка вместо ключа (простая замена по вхождению)
            for e in exps:
                v1 = base.replace(key, e)  # rough replace; в рус. тексте часто норм
                v1n = _normalize_ru(v1)
                if v1n not in seen:
                    variants.append(v1)
                    seen.add(v1n)
            # 2) Обогащение (добавить фразы в конец)
            enrich_tokens.extend(exps)

    if enrich_tokens:
        enriched = base + " (" + ", ".join(sorted(set(enrich_tokens))[:5]) + ")"
        en_norm = _normalize_ru(enriched)
        if en_norm not in seen:
            variants.append(enriched)
            seen.add(en_norm)

    # Отсечь слишком много
    if len(variants) > MAX_QUERY_VARIANTS:
        variants = variants[:MAX_QUERY_VARIANTS]

    return variants

# ===================== Фильтры (NL → Qdrant Filter) =====================
def parse_filters(text: str) -> Tuple[Optional[Filter], str]:
    """Парсит фильтры из естественного текста (дата/месяц/год, sentiment, lang, channel, метрики)."""
    t = text.lower()
    must: List[Any] = []
    ts_gte = ts_lte = None
    now = datetime.utcnow().replace(tzinfo=timezone.utc)

    # 1) относительные диапазоны
    start_date = parse_time_filter(t)
    if start_date:
        ts_gte, ts_lte = int(start_date.timestamp()), int(now.timestamp())

    # 2) явные даты/месяцы/годы и "последние N ..."
    year = next((int(m.group(1)) for m in [RE_YEAR.search(t)] if m), None)
    month = next((m for stem, m in RU_MONTHS.items() if stem in t), None)
    last_period = RE_LAST_N_DAYS.search(t)

    if not ts_gte and last_period:
        n = int(last_period.group(2))
        unit = last_period.group(3).lower()
        start = now - timedelta(days=n) if unit.startswith("дн") else \
                now - timedelta(weeks=n) if unit.startswith("недел") else \
                now - timedelta(days=30*n)
        ts_gte, ts_lte = int(start.timestamp()), int(now.timestamp())
    elif not ts_gte and month:
        if not year:
            year = now.year
        ts_gte = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp())
        next_month = month + 1 if month < 12 else 1
        next_year = year if month < 12 else year + 1
        ts_lte = int((datetime(next_year, next_month, 1, tzinfo=timezone.utc) - timedelta(seconds=1)).timestamp())

    # 3) время
    if ts_gte or ts_lte:
        rng = {}
        if ts_gte: rng["gte"] = ts_gte
        if ts_lte: rng["lte"] = ts_lte
        must.append(FieldCondition(key=DATE_TS_FIELD, range=Range(**rng)))

    # 4) sentiment
    for stem, pair in SENTIMENT_MAP.items():
        if stem in t:
            sent_str, sent_int = pair
            must.append(Filter(should=[
                FieldCondition(key="sentiment", match=MatchValue(value=sent_str)),
                FieldCondition(key="sentiment", match=MatchValue(value=sent_int))
            ]))
            break

    # 5) язык (не обязателен; корпус ru)
    for stem, code in LANG_HINTS.items():
        if stem in t:
            must.append(FieldCondition(key="lang", match=MatchValue(value=code)))
            break

    # 6) канал по ID в тексте
    for rex, key in [(RE_CHANNEL, "channel_id"), (RE_ID, "channel_id")]:
        m = rex.search(t)
        if m:
            must.append(FieldCondition(key=key, match=MatchValue(value=int(m.group(2)))))

    # 7) пороги
    for rex, key in [(RE_REACT_GE, "reaction_score"), (RE_CONF_GE, "confidence")]:
        m = rex.search(t)
        if m:
            must.append(FieldCondition(key=key, range=Range(gte=float(m.group(2).replace(",", ".")))))

    flt = Filter(must=must) if must else None
    cleaned = re.sub(r"\s+", " ", t).strip()
    return flt, cleaned

# ===================== UI → Qdrant Filter =====================
def build_ui_filter(ui: Dict[str, Any]) -> Optional[Filter]:
    """
    Фильтр из UI:
      - date_from/date_to (по DATE_TS_FIELD)
      - sentiment: список [-1,0,1]
      - channels: список channel_id — матч и по int, и по str
    """
    if not ui:
        return None
    must: List[Any] = []

    # Дата
    date_from = ui.get("date_from")
    date_to = ui.get("date_to")
    rng = {}
    if date_from:
        ts = _to_ts_safe(str(date_from))
        if ts is not None:
            rng["gte"] = ts
    if date_to:
        ts = _to_ts_safe(str(date_to), end=True)
        if ts is not None:
            rng["lte"] = ts
    if rng:
        must.append(FieldCondition(key=DATE_TS_FIELD, range=Range(**rng)))

    # Sentiment
    sents = ui.get("sentiment") or []
    if sents:
        must.append(Filter(should=[FieldCondition(key="sentiment", match=MatchValue(value=s)) for s in sents]))

    # Channels — тип-агностично
    chs = ui.get("channels") or []
    if chs:
        should_ch: List[Any] = []
        for cid in chs:
            try:
                cid_int = int(cid)
                should_ch.append(FieldCondition(key="channel_id", match=MatchValue(value=cid_int)))
                should_ch.append(FieldCondition(key="channel_id", match=MatchValue(value=str(cid_int))))
            except Exception:
                pass
            should_ch.append(FieldCondition(key="channel_id", match=MatchValue(value=str(cid))))
        must.append(Filter(should=should_ch))

    return Filter(must=must) if must else None

# ===================== Поиск =====================
def fuse_hits(hits: List[Tuple[int, float, Dict]]) -> List[Tuple[int, float, Dict]]:
    score_map: Dict[int, float] = {}
    payload_map: Dict[int, Dict] = {}
    for doc_id, score, payload in hits:
        if doc_id in score_map:
            score_map[doc_id] += score
        else:
            score_map[doc_id] = score
            payload_map[doc_id] = payload
    fused = [(i, s, payload_map[i]) for i, s in score_map.items() if s >= SCORE_THRESHOLD]
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:FINAL_K]

def build_context(hits: List[Dict[str, Any]]) -> str:
    parts, total = [], 0
    for i, h in enumerate(hits, 1):
        t = h.get("text", "").strip()
        if not t:
            continue
        block = f"[{i}] {t}\n"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block)
    return "".join(parts)

def search_topics_first_multi(client: QdrantClient, embedder: SentenceTransformer, query_variants: List[str], flt: Optional[Filter]) -> List[str]:
    """Ищем похожие темы по нескольким вариантам запроса, объединяем."""
    subjects_set: set[str] = set()
    for q in query_variants[:3]:  # достаточно первых 2-3 вариантов
        qvec = encode(embedder, q)
        try:
            hits = client.search(
                collection_name="topics_summary",
                query_vector=("text", qvec),
                with_payload=True,
                limit=5,
                query_filter=flt,
            )
            for h in hits:
                payload = h.payload or {}
                subj = payload.get("subject") or payload.get("topic")
                if subj:
                    subjects_set.add(subj)
        except Exception as e:
            print(f"⚠️ Ошибка при поиске тем: {e}")
    return list(subjects_set)

# ===================== Генерация ответа =====================
def generate_answer_gigachat(context: str, user_query: str) -> str:
    if not USE_OPENAI or openai_client is None:
        return "Не настроен доступ к GigaChat (API_KEY). Пожалуйста, укажи API_KEY в окружении."
    messages = [
        {"role": "system", "content": PROMPT_SYSTEM},
        {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {user_query}"}
    ]
    response = openai_client.chat.completions.create(
        model="GigaChat/GigaChat-2-Max",
        messages=messages,
        max_tokens=2500,
        temperature=0.5,
        top_p=0.95,
        presence_penalty=0
    )
    return response.choices[0].message.content

# ===================== RAG ответ =====================
def rag_answer(query_text: str, ui_filters: Optional[Dict[str, Any]] = None) -> dict:
    client, embedder = init_clients()

    # 1) Фильтры: из текста и из UI
    flt_from_text, cleaned_query = parse_filters(query_text)
    flt_from_ui = build_ui_filter(ui_filters or {})
    user_filter = Filter(must=[flt for flt in [flt_from_text, flt_from_ui] if flt]) if (flt_from_text or flt_from_ui) else None

    # 2) Переписанные варианты запроса
    query_variants = expand_query_variants(query_text if query_text else cleaned_query)
    if cleaned_query and cleaned_query != _normalize_ru(query_text):
        # иногда полезно положить и "очищенный" вариант
        query_variants.append(cleaned_query)
    # Уникализируем, ограничиваем
    uq = []
    seen = set()
    for v in query_variants:
        vn = _normalize_ru(v)
        if vn not in seen:
            uq.append(v)
            seen.add(vn)
    query_variants = uq[:MAX_QUERY_VARIANTS]

    # 3) Темы по нескольким вариантам
    subjects = search_topics_first_multi(client, embedder, query_variants, user_filter)
    use_subject_filter = bool(subjects)
    if use_subject_filter:
        subject_filter_obj = Filter(should=[FieldCondition(key="subject", match=MatchValue(value=s)) for s in subjects])
        base_filter = Filter(must=[f for f in [user_filter, subject_filter_obj] if f])
    else:
        base_filter = user_filter

    # 4) Поиск постов по каждому варианту запроса и каждой "голове" векторов
    hits: List[Dict[str, Any]] = []

    def _search_once(vec_name: str, qtext: str, qfilter: Optional[Filter]):
        try:
            qvec = encode(embedder, qtext)
            res = client.search(
                collection_name="final_db",
                query_vector=(vec_name, qvec),
                with_payload=True,
                limit=TOP_K_EACH,
                query_filter=qfilter,
            )
            local = []
            for h in res:
                local.append({
                    "id": int(h.id),
                    "score": float(h.score),
                    "text": (h.payload or {}).get("text") or (h.payload or {}).get("example_post") or "",
                    "payload": h.payload or {},
                    "vector_name": vec_name,
                })
            return local
        except Exception as e:
            print(f"⚠️ Ошибка поиска ({vec_name}) для варианта '{qtext[:40]}...': {e}")
            return []

    for qv in query_variants:
        hits.extend(_search_once("text", qv, base_filter))
        hits.extend(_search_once("meta", qv, base_filter))

    # 5) Fallback: если с subject пусто — пробуем без него
    if use_subject_filter and len(hits) < 3:
        hits = []
        for qv in query_variants:
            hits.extend(_search_once("text", qv, user_filter))
            hits.extend(_search_once("meta", qv, user_filter))

    # 5.1) Доп. fallback: ослабить только sentiment (оставить канал и дату)
    if not hits and ui_filters:
        ui_copy = dict(ui_filters)
        ui_copy.pop("sentiment", None)
        loose_ui = build_ui_filter(ui_copy)
        loose_filter = Filter(must=[flt for flt in [flt_from_text, loose_ui] if flt]) if (flt_from_text or loose_ui) else None
        for qv in query_variants:
            hits.extend(_search_once("text", qv, loose_filter))
            hits.extend(_search_once("meta", qv, loose_filter))

    # 6) Взвешивание/слияние
    text_hits = [h for h in hits if h["vector_name"] == "text"]
    meta_hits = [h for h in hits if h["vector_name"] == "meta"]
    fused = fuse_hits(
        [(h["id"], h["score"] * W_TEXT, h["payload"]) for h in text_hits] +
        [(h["id"], h["score"] * W_META, h["payload"]) for h in meta_hits]
    )

    final_hits: List[Dict[str, Any]] = []
    for doc_id, score, payload in fused:
        final_hits.append({
            "id": doc_id,
            "score": score,
            "text": payload.get("text") or payload.get("example_post") or "",
            "payload": payload
        })

    if not final_hits:
        return {
            "answer": "По текущим данным ничего не найдено.",
            "sources": [],
            "topics_found": subjects,
            "rewrites": query_variants
        }

    # 7) Контекст → генерация
    context = build_context(final_hits)
    answer_text = generate_answer_gigachat(context, query_text)

    
    # === Справочник каналов ===
    CHANNELS = {
        1101170442: "rian_ru",
        1222869173: "naebnet",
        1319248631: "whackdoor",
        1099860397: "rbc_news",
        2005877458: "ecnomica",
        2416194304: "ecotopor",
        1307778786: "exploitex",
        1158411788: "htech_plus",
        1708761316: "novosti_efir",
        2497181539: "technomedia",
        1867803460: "TechnoMedi",
        1794988016: "GPTMainNews",
        1058912111: "rhymestg",
        2499221807: "aivengonews",
        1006147755: "mudak",
        1158411788: "htech_plus",
    }

    # Источники
    sources: List[Dict[str, Any]] = []
    for i, h in enumerate(final_hits, 1):
        p = h["payload"]
        channel_id = p.get("channel_id")
        message_id = p.get("message_id")
        username = CHANNELS.get(channel_id)
        link = f"https://t.me/{username}/{message_id}" if username and message_id else None

        sources.append({
            "rank": i,
            "score": round(h["score"], 4),
            "message_id": message_id,
            "channel_id": channel_id,
            "channel_username": username,
            "date": p.get("date"),
            "sentiment": p.get("sentiment"),
            "confidence": p.get("confidence"),
            "reaction_score": p.get("reaction_score"),  # ← правильно
            "social_reactions": p.get("reaction_score"), # ← чтобы UI видел
            "subject": p.get("subject"),
            "preview": h["text"],
            "link": link
        })

    return {
        "answer": answer_text,
        "sources": sources,
        "topics_found": subjects,
        "rewrites": query_variants
    }


# ===================== CLI (опционально) =====================
if __name__ == "__main__":
    print(f"RAG NL Filters (collections={'+'.join(COLLECTIONS)}, embedder={EMBEDDER_NAME})")
    q = input("Запрос: ").strip()
    out = rag_answer(q, ui_filters={})
    print("\n=== REWRITES ===")
    for i, v in enumerate(out.get("rewrites", []), 1):
        print(f"{i}. {v}")
    print("\n=== ANSWER ===\n", out["answer"])
    print("\n=== SOURCES ===")
    for s in out["sources"]:
        print(s)

