# news_matcher.py
# -*- coding: utf-8 -*-
"""
Мэтчинг одной и той же новости между разными Telegram-каналами.
Зависит от rag_core.py (init_clients, encode, parse_filters, build_ui_filter, search_topics_first_multi, fuse_hits).

Usage (CLI):
    python news_matcher.py "ИИ новости за неделю"
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from sentence_transformers import SentenceTransformer

# ---- импортируем части из твоего RAG ----
from rag_core import (
    init_clients, encode,
    parse_filters, build_ui_filter, search_topics_first_multi, fuse_hits,
    MAX_QUERY_VARIANTS, W_TEXT, W_META
)

# ===================== Параметры мэтчинга =====================
DATE_TS_FIELD = "ts"          # unix timestamp в payload (как в rag_core через ENV)
DUP_WINDOW_HOURS = 48         # окно по времени для кандидатов
DUP_CANDIDATES_LIMIT = 200    # сколько брать кандидатов из Qdrant
DUP_COS_THR = 0.78            # порог cosine для «тот же смысл»
DUP_SCORE_THR = 0.72          # порог интегрального скора

# ===================== Вспомогательные функции =====================
_URL_RE = re.compile(r'https?://\S+', re.I)

def extract_urls_and_domains(text: str) -> Tuple[List[str], List[str]]:
    urls = _URL_RE.findall(text or "")
    domains: List[str] = []
    for u in urls:
        try:
            d = urlparse(u).netloc.lower()
            if d.startswith("www."):
                d = d[4:]
            domains.append(d)
        except Exception:
            pass
    return sorted(set(urls)), sorted(set(domains))

def jaccard_trigrams(a: str, b: str) -> float:
    def _trigrams(s: str):
        toks = re.findall(r'\w+', (s or "").lower(), flags=re.U)
        return set(zip(toks, toks[1:], toks[2:])) if len(toks) >= 3 else set()
    A, B = _trigrams(a), _trigrams(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def url_domain_overlap(dom_a: List[str], dom_b: List[str]) -> float:
    if not dom_a or not dom_b:
        return 0.0
    A, B = set(dom_a), set(dom_b)
    return len(A & B) / len(A | B)

def match_score(cos: float, jac: float, urlov: float) -> float:
    # Базовые веса: семантика > лексика > домены
    return 0.6 * float(cos) + 0.3 * float(jac) + 0.1 * float(urlov)

def _payload_ts(payload: Dict[str, Any]) -> Optional[int]:
    v = payload.get(DATE_TS_FIELD)
    try:
        return int(v) if v is not None else None
    except Exception:
        return None

# ===================== Поиск дублей для одного поста =====================
def find_duplicates_for_hit(
    client: QdrantClient,
    embedder: SentenceTransformer,
    hit: Dict[str, Any],
    window_hours: int = DUP_WINDOW_HOURS,
    limit: int = DUP_CANDIDATES_LIMIT,
) -> List[Dict[str, Any]]:
    """
    Возвращает список дублей (похожие по событию посты в других каналах).
    Элементы: {id, score_final, score_cosine, score_jaccard, text, payload}
    """
    payload = hit.get("payload", {}) or {}
    seed_text = hit.get("text") or payload.get("text") or payload.get("example_post") or ""
    seed_vec = encode(embedder, seed_text)
    seed_ts = _payload_ts(payload)
    seed_ch = payload.get("channel_id")

    if seed_ts is None:
        return []

    t_min = seed_ts - window_hours * 3600
    t_max = seed_ts + window_hours * 3600

    try:
        cand_hits = client.search(
            collection_name="final_db",
            query_vector=("text", seed_vec),
            with_payload=True,
            limit=limit,
            query_filter=Filter(must=[FieldCondition(key=DATE_TS_FIELD, range=Range(gte=t_min, lte=t_max))]),
        )
    except Exception as e:
        print(f"⚠️ Кандидаты для дублей: {e}")
        return []

    seed_urls, seed_domains = extract_urls_and_domains(seed_text)
    if not seed_domains:
        seed_domains = (payload.get("domains") or [])

    dup_list: List[Dict[str, Any]] = []
    for ch in cand_hits:
        p = ch.payload or {}

        # Пропускаем сам seed-пост (если channel_id/message_id совпали)
        if seed_ch is not None and p.get("channel_id") == seed_ch and p.get("message_id") == payload.get("message_id"):
            continue

        cand_text = p.get("text") or p.get("example_post") or ""
        cand_domains = p.get("domains") or extract_urls_and_domains(cand_text)[1]

        cos = float(ch.score)  # при Distance=Cosine — это similarity
        jac = jaccard_trigrams(seed_text, cand_text)
        uov = url_domain_overlap(seed_domains, cand_domains)
        sc = match_score(cos, jac, uov)

        if (cos >= DUP_COS_THR) or (sc >= DUP_SCORE_THR):
            dup_list.append({
                "id": int(ch.id),
                "score_cosine": round(cos, 4),
                "score_jaccard": round(jac, 4),
                "score_final": round(sc, 4),
                "text": cand_text,
                "payload": p,
            })

    dup_list.sort(key=lambda x: (x["score_final"], x["score_cosine"]), reverse=True)
    return dup_list

# ===================== Публичная функция для UI/API =====================
def rag_match_across_channels(
    query_text: str,
    ui_filters: Optional[Dict[str, Any]] = None,
    pick_index: int = 0,
) -> dict:
    """
    1) Делает поиск как в rag_core (без генерации)
    2) Берёт один seed-пост (по умолчанию лучший)
    3) Находит его дублеты в других каналах
    """
    client, embedder = init_clients()

    # 1) Фильтры из запроса и UI
    flt_from_text, cleaned_query = parse_filters(query_text)
    flt_from_ui = build_ui_filter(ui_filters or {})
    user_filter = Filter(must=[flt for flt in [flt_from_text, flt_from_ui] if flt]) if (flt_from_text or flt_from_ui) else None

    # 2) Переписанные варианты (как в RAG)
    from rag_core import _normalize_ru, expand_query_variants  # локальный импорт, чтобы избежать циклов при типизации
    query_variants = expand_query_variants(query_text if query_text else cleaned_query)
    if cleaned_query and cleaned_query != _normalize_ru(query_text):
        query_variants.append(cleaned_query)
    uq, seen = [], set()
    for v in query_variants:
        vn = _normalize_ru(v)
        if vn not in seen:
            uq.append(v); seen.add(vn)
    query_variants = uq[:MAX_QUERY_VARIANTS]

    # 3) Темы → optional subject-фильтр
    subjects = search_topics_first_multi(client, embedder, query_variants, user_filter)
    use_subject_filter = bool(subjects)
    if use_subject_filter:
        subject_filter_obj = Filter(should=[FieldCondition(key="subject", match=MatchValue(value=s)) for s in subjects])
        base_filter = Filter(must=[f for f in [user_filter, subject_filter_obj] if f])
    else:
        base_filter = user_filter

    # 4) Поиск хитов по вариантам и двум «головам» вектора
    def _search_once(vec_name: str, qtext: str, qfilter: Optional[Filter]):
        try:
            qvec = encode(embedder, qtext)
            res = client.search(
                collection_name="final_db",
                query_vector=(vec_name, qvec),
                with_payload=True,
                limit=8,  # TOP_K_EACH из rag_core
                query_filter=qfilter,
            )
            local = []
            for h in res:
                payload = h.payload or {}
                local.append({
                    "id": int(h.id),
                    "score": float(h.score),
                    "text": payload.get("text") or payload.get("example_post") or "",
                    "payload": payload,
                    "vector_name": vec_name,
                })
            return local
        except Exception as e:
            print(f"⚠️ Ошибка поиска ({vec_name}): {e}")
            return []

    hits: List[Dict[str, Any]] = []
    for qv in query_variants:
        hits.extend(_search_once("text", qv, base_filter))
        hits.extend(_search_once("meta", qv, base_filter))

    if use_subject_filter and len(hits) < 3:
        hits = []
        for qv in query_variants:
            hits.extend(_search_once("text", qv, user_filter))
            hits.extend(_search_once("meta", qv, user_filter))

    # 5) Слияние как в rag_core
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
            "seed": None,
            "duplicates": [],
            "topics_found": subjects,
            "rewrites": query_variants,
            "message": "Ничего не найдено."
        }

    # 6) Выбираем seed и ищем дублеты
    seed_index = min(max(pick_index, 0), len(final_hits) - 1)
    seed = final_hits[seed_index]
    duplicates = find_duplicates_for_hit(client, embedder, seed, window_hours=DUP_WINDOW_HOURS, limit=DUP_CANDIDATES_LIMIT)

    # 7) Декорируем ссылками (как в rag_core)
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

    def _make_link(p):
        ch_id = p.get("channel_id")
        mid = p.get("message_id")
        uname = CHANNELS.get(ch_id)
        return f"https://t.me/{uname}/{mid}" if uname and mid else None

    seed_view = {
        "id": seed["id"],
        "score": round(seed["score"], 4),
        "channel_id": seed["payload"].get("channel_id"),
        "message_id": seed["payload"].get("message_id"),
        "link": _make_link(seed["payload"]),
        "text": seed["text"],
        "payload": seed["payload"],
    }

    dups_view = []
    for d in duplicates:
        p = d["payload"]
        dups_view.append({
            "id": d["id"],
            "score_final": d["score_final"],
            "score_cosine": d["score_cosine"],
            "score_jaccard": d["score_jaccard"],
            "channel_id": p.get("channel_id"),
            "message_id": p.get("message_id"),
            "link": _make_link(p),
            "text": d["text"],
            "payload": p,
        })

    return {
        "seed": seed_view,
        "duplicates": dups_view,
        "topics_found": subjects,
        "rewrites": query_variants
    }

# ===================== CLI =====================
if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]).strip() or "ИИ новости за неделю"
    out = rag_match_across_channels(q, ui_filters=None)
    seed = out.get("seed")
    print("=== SEED ===")
    if seed:
        print(seed.get("link"), "|", seed.get("text")[:120].replace("\n", " "))
    else:
        print("Seed не найден")
    print("\n=== DUPLICATES (top 10) ===")
    for d in out.get("duplicates", [])[:10]:
        print(d["score_final"], d.get("link"), "|", (d.get("text") or "")[:100].replace("\n", " "))
