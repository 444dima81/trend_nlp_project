# main.py
# -*- coding: utf-8 -*-
import os
from datetime import date, datetime, timezone, timedelta
from typing import Any, Dict, List
import xlsxwriter
import streamlit as st

# === ваш RAG-модуль ===
from rag_core import (
    rag_answer,                # основной пайплайн ответа
    parse_filters,             # парсер NL-фильтров
    init_clients,              # инициализация Qdrant + эмбеддер
    Filter, FieldCondition, MatchValue, Range
)

# === модуль мэтчинга дублей ===
from news_matcher import rag_match_across_channels

# -----------------------------------------------------------------------------
# Конфиг страницы
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Multi-Serach News RAG · Telegram", page_icon="🗞️", layout="wide")

# --- Sidebar навигация ---
st.sidebar.title("Навигация")
page = st.sidebar.radio(
    "Перейти к:",
    ["🔍 Поиск новостей", "📊 Аналитика"]
)

# ---------- СТИЛЬ (тёмный минимализм в духе macOS) ----------
STYLES = """
<style>
html, body, [class*="css"]  {
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue",
               Helvetica, Arial, "Segoe UI", Roboto, sans-serif;
}

:root {
  --bg: #0E0E11;
  --bg-2: #141418;
  --text: #EDEDED;
  --muted: #A9ABB3;
  --accent: #8B5CF6;
  --pill: #1B1B22;
  --border: #23232B;
}

body { background: var(--bg); color: var(--text); }
section.main > div { padding-top: 0.6rem; }

.header-wrap {
  display:flex; align-items:center; justify-content:space-between;
  padding: 8px 0 6px 0; border-bottom: 1px solid var(--border);
}

.pills { display:flex; gap:8px; flex-wrap:wrap; }
.pill {
  background: var(--pill); border: 1px solid var(--border);
  padding: 4px 10px; border-radius: 999px; color: var(--muted);
  font-size: 12.5px;
}

.card {
  border: 1px solid var(--border); background: var(--bg-2);
  border-radius: 12px; padding: 12px 14px; margin-bottom: 10px;
}
.card small { color: var(--muted); }
.card .meta { display:flex; gap:8px; flex-wrap:wrap; margin-top:6px; }
.badge { background:#1e1e26; border:1px solid var(--border);
  padding: 2px 8px; border-radius: 999px; font-size:12px; color:#cfd1d8; }

.superset-frame { border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
a { color: #c7b7ff; text-decoration: none; }
a:hover { text-decoration: underline; }
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)

# ---------- КАНАЛЫ: id → display_name (дополни своими) ----------
CHANNEL_NAMES: Dict[int, str] = {
    1101170442: "РИА Новости",
    1222869173: "NN",
    1319248631: "Бэкдор",
    1099860397: "РБК. Новости. Главное",
    2005877458: "The Экономист",
    2416194304: "Топор. Экономика.",
    1307778786: "Эксплойт",
    1158411788: "Хайтек+",
    1708761316: "Прямой Эфир • Новости",
    2497181539: "Техночат",
    1867803460: "TechnoMedi",
    1794988016: "GPTMain News",
    1058912111: "Рифмы и Панчи 🤯",
    2499221807: "ИИ by AIvengo",
    1006147755: "MDK",
}
CHANNEL_NAME_TO_ID: Dict[str, int] = {v: k for k, v in CHANNEL_NAMES.items()}

# -----------------------------------------------------------------------------
# Сессия
# -----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, Any]] = []
if "active_filters" not in st.session_state:
    st.session_state.active_filters = {"date_from": None, "date_to": None, "sentiment": [], "channels": []}
if "superset_iframe" not in st.session_state:
    st.session_state.superset_iframe = ""
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_sources" not in st.session_state:
    st.session_state.last_sources: List[Dict[str, Any]] = []

# -----------------------------------------------------------------------------
# Кэш ресурсов
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_clients():
    return init_clients()  # (QdrantClient, SentenceTransformer)

# -----------------------------------------------------------------------------
# Утилиты UI
# -----------------------------------------------------------------------------
def __to_ts(iso_str: str, end: bool = False) -> int:
    dt = datetime.fromisoformat(iso_str)
    if end:
        dt = dt.replace(hour=23, minute=59, second=59)
    return int(dt.replace(tzinfo=timezone.utc).timestamp())

def render_filter_pills(filters: Dict[str, Any]):
    pills = []
    if filters.get("date_from") or filters.get("date_to"):
        pills.append(f"date_utc: {filters.get('date_from') or '…'} → {filters.get('date_to') or '…'}")
    if filters.get("sentiment"):
        pills.append("sentiment: " + ",".join(map(str, filters["sentiment"])))
    if filters.get("channels"):
        names = []
        for cid in filters["channels"]:
            try:
                names.append(CHANNEL_NAMES.get(int(cid), str(cid)))
            except Exception:
                names.append(str(cid))
        pills.append("channels: " + ", ".join(names))
    if pills:
        st.markdown('<div class="pills">' + "".join([f'<span class="pill">{p}</span>' for p in pills]) + '</div>', unsafe_allow_html=True)

def apply_filters_ui():
    df, dt = st.date_input(
        "Диапазон дат (date_utc)",
        value=(date.today().replace(month=1, day=1), date.today()),
        key="f_dates"
    )

    sentiments = st.multiselect(
        "Sentiment",
        options=[-1, 0, 1],
        default=st.session_state.active_filters.get("sentiment", [])
    )

    selected_names = st.multiselect(
        "Каналы (по названию)",
        options=sorted(CHANNEL_NAME_TO_ID.keys()),
        default=[CHANNEL_NAMES.get(cid) for cid in st.session_state.active_filters.get("channels", []) if CHANNEL_NAMES.get(cid)],
        placeholder="Выбери один или несколько каналов"
    )
    selected_ids = [CHANNEL_NAME_TO_ID[name] for name in selected_names]

    if st.button("Применить фильтры", use_container_width=True):
        st.session_state.active_filters.update({
            "date_from": str(df),
            "date_to": str(dt),
            "sentiment": sentiments,
            "channels": selected_ids,
        })
        st.toast("Фильтры применены")

def embed_superset(iframe_src: str, height: int = 560):
    st.markdown(
        f'<div class="superset-frame"><iframe src="{iframe_src}" width="100%" height="{height}" frameborder="0"></iframe></div>',
        unsafe_allow_html=True
    )

def render_sources(sources: List[Dict[str, Any]]):
    """Карточки результатов обычного поиска (без кнопок дублей — теперь дубликаты ищем отдельно)."""
    for s in sources:
        channel_id = s.get("channel_id")
        try:
            channel_name = CHANNEL_NAMES.get(int(channel_id), str(channel_id)) if channel_id is not None else "—"
        except Exception:
            channel_name = str(channel_id or "—")

        link_html = f'<div style="margin-top:6px;"><a href="{s["link"]}" target="_blank">🔗 Открыть в Telegram</a></div>' if s.get("link") else ""

        conf = s.get("confidence")
        conf_display = f"{conf:.4f}" if isinstance(conf, (float, int)) else (conf or "—")
        score = s.get("score")
        score_display = f"{score:.4f}" if isinstance(score, (float, int)) else (score or "—")
        react = s.get("reaction_score") or s.get("social_reactions") or "—"

        st.markdown(
            f"""
            <div class="card">
                <div><b>{s.get("title") or f'Сообщение {s.get("message_id","")}'}</b></div>
                <div style="margin-top:4px">{s.get("preview","Фрагмент...")}</div>
                <div class="meta" style="margin-top:8px">
                    <span class="badge">{s.get("date","—")}</span>
                    <span class="badge">channel: {channel_name}</span>
                    <span class="badge">sent: {s.get("sentiment","—")}</span>
                    <span class="badge">conf: {conf_display}</span>
                    <span class="badge">score: {score_display}</span>
                    <span class="badge">reactions: {react}</span>
                </div>
                {link_html}
            </div>
            """,
            unsafe_allow_html=True
        )

def trend_search(filters: Dict[str, Any], limit: int = 8) -> List[Dict[str, Any]]:
    """Простой тренд-поиск по коллекции 'topics_summary'."""
    client, _ = get_clients()

    must = []
    if filters.get("date_from") or filters.get("date_to"):
        rng = {}
        if filters.get("date_from"):
            rng["gte"] = int(__to_ts(filters["date_from"]))
        if filters.get("date_to"):
            rng["lte"] = int(__to_ts(filters["date_to"], end=True))
        if rng:
            must.append(FieldCondition(key="ts", range=Range(**rng)))

    if filters.get("sentiment"):
        should = [FieldCondition(key="sentiment", match=MatchValue(value=s)) for s in filters["sentiment"]]
        must.append(Filter(should=should))

    if filters.get("channels"):
        should = [FieldCondition(key="channel_id", match=MatchValue(value=int(cid))) for cid in filters["channels"]]
        must.append(Filter(should=should))

    qf = Filter(must=must) if must else None

    topics_count: Dict[str, int] = {}
    next_offset = None
    total_seen = 0
    try:
        while True and total_seen < 1000:
            points, next_offset = client.scroll(
                collection_name="topics_summary",
                limit=200,
                with_payload=True,
                offset=next_offset,
                scroll_filter=qf
            )
            if not points:
                break
            for p in points:
                subj = (p.payload or {}).get("subject") or (p.payload or {}).get("topic")
                if not subj:
                    continue
                topics_count[subj] = topics_count.get(subj, 0) + 1
                total_seen += 1
            if not next_offset:
                break
    except Exception as e:
        st.warning(f"Не удалось получить тренды: {e}")

    items = sorted(topics_count.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    return [{"title": t, "preview": f"Упоминаний: {c}", "date": "—", "sentiment": "—", "score": c} for t, c in items]


if page == "🔍 Поиск новостей":
    # -----------------------------------------------------------------------------
    # Хедер
    # -----------------------------------------------------------------------------
    st.markdown(
        '<div class="header-wrap"><h3 style="margin:0">Multi-Serach 🗞️ News RAG · Telegram</h3>'
        '<div style="color:#A9ABB3; font-size:13px; margin-top:2px">'
        'Semantic news search with citations & filters</div></div>',
        unsafe_allow_html=True
    )
    st.write("")

    # -----------------------------------------------------------------------------
    # Левая/правая колонка
    # -----------------------------------------------------------------------------
    left, right = st.columns([0.7, 0.3], gap="large")

    # Левая колонка — история и ввод запроса
    with left:
        for item in st.session_state.history:
            if item["role"] == "user":
                with st.chat_message("user"):
                    st.write(item["content"])
                    render_filter_pills(item.get("filters", {}))
            else:
                with st.chat_message("assistant"):
                    st.write(item.get("content", ""))
                    if "sources" in item:
                        render_sources(item.get("sources", []))

        user_msg = st.chat_input("Поиск по новостям Telegram… Например: «положительные новости за июль 2025»")
        if user_msg:
            with st.spinner("Ищем и собираем ответ…"):
                result = rag_answer(user_msg, ui_filters=st.session_state.active_filters)


            answer = result.get("answer", "—")
            sources = result.get("sources", [])

            # Фильтрация по каналам
            sel_channels = st.session_state.active_filters.get("channels") or []
            if sel_channels:
                sel_channels_set = {int(x) for x in sel_channels}
                sources = [s for s in sources if s.get("channel_id") and int(s["channel_id"]) in sel_channels_set]

            # Сохраняем для блока дублей
            st.session_state.last_query = user_msg
            st.session_state.last_sources = sources

            # Парсинг фильтров
            qfilter, cleaned = parse_filters(user_msg)
            parsed_filters = st.session_state.active_filters.copy()
            parsed_filters.update({"__parsed": cleaned})

            st.session_state.history += [
                {"role": "user", "content": user_msg, "filters": parsed_filters},
                {"role": "assistant", "content": answer, "sources": sources}
            ]
            st.rerun()


    with right:
        # Инструкция
    # === Добавлено описание и инструкция ===
        with st.expander("ℹ️ Как работает поиск:", expanded=False):
            st.markdown("""
            <div style="margin:6px 0 10px 0; padding:14px 18px; border-radius:12px;
                        background:var(--bg-2); border:1px solid var(--border); color:var(--muted);
                        font-size:14px; line-height:1.55;">
            <b>⚠️ Внимание:</b><br>
            RAG-поиск может ошибаться — всегда проверяйте первоисточник.<br><br>

            <b>🧠 Как работает поиск:</b><br>
            Используется <b>RAG-модель</b>, которая извлекает релевантные Telegram-посты 
            из векторной базы данных и формирует краткий ответ по смыслу, а не по простому совпадению слов.<br><br>

            <b>🔍 Примеры запросов:</b><br>
            • <i>Позитивные новости об экономике за сентябрь</i><br>
            • <i>Новости про ИИ за прошлую неделю</i><br>
            • <i>Негативные новости про недвижимость за последние полгода</i><br><br>

            <b>💡 Рекомендации:</b><br>
            • Тональность (позитив/негатив) — субъективна. Сначала ищите без семантической фильтрации, 
            затем применяйте фильтр по тону.<br>
            • Можно уточнять время: <i>«за март 2025»</i> или <i>«за последние 3 месяца»</i>.<br>
            • Если ничего не найдено — попробуйте переформулировать или сделать запрос шире.<br><br>

            <b>📊 Что означают метрики:</b><br>
            • <b>channel</b> — канал, из которого взят пост.<br>
            • <b>sent</b> — тональность: <code>-1</code> (негатив), <code>0</code> (нейтрально), <code>1</code> (позитив).<br>
            • <b>conf</b> — уверенность модели в оценке тональности (0–1). Чем выше, тем надёжнее прогноз.<br>
            • <b>score</b> — релевантность поста вашему запросу (0–1).<br>
            • <b>reactions</b> — эмоциональный отклик аудитории (от <code>-1</code> до <code>1</code>): 
            ближе к 1 — позитивные реакции, ближе к -1 — негативные.<br><br>

            Используйте эти показатели, чтобы оценить не только содержание, но и достоверность, 
            уверенность модели и реакцию аудитории.
            </div>

            """, unsafe_allow_html=True)

            

        # Фильтры
        with st.expander("📅 Добавить фильтр для текущего поиска"):
            apply_filters_ui()

        st.markdown("---")
        st.subheader("⚡️ Управление")

        # Перенесенные кнопки из хедера
        if st.button("📈 Trend search", use_container_width=True):
            items = trend_search(st.session_state.active_filters)
            st.session_state.history.append({"role": "assistant", "type": "trends", "content": "Тренды", "sources": items})

        if st.button("🧹 Очистить чат", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_query = ""
            st.session_state.last_sources = []
            st.session_state.just_cleared = True
            st.rerun()  # ← вот эта строка решает проблему «со второго раза»

        if st.session_state.pop("just_cleared", False):
            st.toast("Чат очищен")

        st.markdown("---")
        st.subheader("🔁 Найти дубликаты по текущей выдаче")
        srcs = st.session_state.get("last_sources") or []

        if not srcs:
            st.caption("Сначала выполните обычный поиск в левой колонке.")
        else:
            def _label(i, s):
                ch = s.get("channel_id")
                try:
                    ch_name = CHANNEL_NAMES.get(int(ch), str(ch))
                except Exception:
                    ch_name = str(ch or "—")
                title = s.get("title") or f"Сообщение {s.get('message_id','')}"
                return f"#{i+1} · {ch_name} · {title}"

            seed_idx = st.selectbox(
                "Выберите пост-«ядро» для поиска дублей:",
                options=list(range(len(srcs))),
                format_func=lambda i: _label(i, srcs[i]),
                index=0
            )

            if st.button("🔎 Найти дубликаты", use_container_width=True):
                with st.spinner("Ищем похожие публикации в других каналах..."):
                    out = rag_match_across_channels(
                        st.session_state.get("last_query", ""),
                        ui_filters=st.session_state.active_filters,
                        pick_index=seed_idx
                    )
                    dups = out.get("duplicates", []) or []

                if not dups:
                    st.info("Дубликаты не найдены.")
                else:
                    for d in dups:
                        p = d.get("payload", {}) or {}
                        ch_id = p.get("channel_id")
                        try:
                            ch_name = CHANNEL_NAMES.get(int(ch_id), str(ch_id))
                        except Exception:
                            ch_name = str(ch_id or "—")
                        link = d.get("link") or "#"
                        preview = (d.get("text") or "")[:300].replace("\n", " ")
                        st.markdown(
                            f"""
                            <div class="card">
                                <div><b>{ch_name}</b> • <a href="{link}" target="_blank">Открыть в Telegram</a></div>
                                <div style="margin-top:6px">{preview}</div>
                                <div class="meta" style="margin-top:8px">
                                    <span class="badge">score: {d.get('score_final')}</span>
                                    <span class="badge">cos: {d.get('score_cosine')}</span>
                                    <span class="badge">jaccard: {d.get('score_jaccard')}</span>
                                    <span class="badge">url: {d.get('score_url_overlap')}</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

        st.markdown("---")
        st.subheader("💾 Сохранение результатов поиска")

        if srcs:
            import pandas as pd
            import io

            df = pd.DataFrame(srcs)
            df = df.rename(columns={
                "title": "Заголовок",
                "text": "Текст",
                "preview": "Превью",
                "channel_id": "ID канала",
                "date": "Дата",
                "sentiment": "Тональность",
                "score": "Score",
                "confidence": "Confidence",
                "reaction_score": "Reactions",
                "link": "Ссылка"
            })

            c1, c2 = st.columns(2)
            with c1:
                csv_data = df.to_csv(index=False)
                st.download_button("💾 Скачать CSV", csv_data, file_name="news.csv", mime="text/csv")
            with c2:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name="News")
                st.download_button(
                    "💾 Скачать Excel",
                    data=excel_buffer.getvalue(),
                    file_name="news.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.info("Сначала выполните поиск, чтобы сохранить результаты.")



 # --- Страница аналитики ---
elif page == "📊 Аналитика":
    import news_analysis
    news_analysis.show_page()       

    # st.subheader("📊 Аналитика (Superset)")
    # sup_url = st.text_input("Superset iframe URL (опционально)", value=st.session_state.superset_iframe, placeholder="https://...")
    # if sup_url != st.session_state.superset_iframe:
    #     st.session_state.superset_iframe = sup_url
    # if st.session_state.superset_iframe:
    #     embed_superset(st.session_state.superset_iframe, height=560)
    # else:
    #     st.info("Тут появятся графики из Superset. Поддерживается iframe / Guest Token embedding.")
