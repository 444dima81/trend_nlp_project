# news_analysis.py
# ------------------------------------------------------
# Упрощённая версия: путь к CSV задан в коде.
# Без загрузчика, с минимальным UI и облаком слов (кириллица OK).
# ------------------------------------------------------

import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from collections import Counter
from itertools import combinations

# Опционально: wordcloud
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

# === Маппинг каналов: channel_id -> display_name ===
CHANNELS = {
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
CHANNELS_STR = {str(k): v for k, v in CHANNELS.items()}


def _wc_font_path() -> str:
    """Возвращает путь к шрифту с кириллицей внутри контейнера."""
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Debian/Ubuntu (fonts-dejavu-core)
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",  # альтернатива
    ):
        if os.path.exists(p):
            return p
    return ""  # WordCloud возьмёт дефолтный шрифт, если ничего не нашли


def render_wordcloud_from_freq(freq: dict, title: str = "Облако слов"):
    """Рендерит облако слов по частотному словарю."""
    if not WORDCLOUD_AVAILABLE:
        st.info("WordCloud не установлен в окружении.")
        return
    if not freq:
        st.warning("Нет данных для облака слов.")
        return

    font_path = _wc_font_path()
    try:
        wc = WordCloud(
            width=1200,
            height=500,
            background_color="white",
            max_words=400,
            collocations=False,
            regexp=r"[А-Яа-яA-Za-z0-9_#@\-]+",
            font_path=font_path or None,
        ).generate_from_frequencies(freq)

        fig = plt.figure(figsize=(12, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        st.subheader(title)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception as e:
        st.error(f"Не удалось построить WordCloud: {e}")


def show_page():
    st.title("📊 Глубокая сводка по CSV")

    # Путь внутри контейнера (смонтирован как ./streamit/data:/app/data:ro)
    CSV_PATH = "/app/data/db.csv"

    # === Загружаем данные ===
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        st.error(f"Не удалось прочитать файл по пути: {CSV_PATH}\n\nОшибка: {e}")
        st.stop()

    st.write("Размер:", df.shape)
    if df is None or df.empty:
        st.info("Файл пуст или не загружен.")
        st.stop()

    # Нормализуем названия колонок
    df.columns = [str(c).strip() for c in df.columns]
    cols_norm = {c: c.lower() for c in df.columns}
    df.rename(columns=cols_norm, inplace=True)

    # Кандидаты колонок
    DATE_CANDIDATES = ["data_utc", "date_utc", "date", "timestamp, time".split(", ")]
    if isinstance(DATE_CANDIDATES[0], list):  # страховка от случайной запятой
        DATE_CANDIDATES = ["data_utc", "date_utc", "date", "timestamp", "time"]
    ID_COLS = ["message_id", "id", "post_id"]
    TEXT_COLS = ["text", "text_clean", "content"]
    KWORDS_COLS = ["k_words", "keywords", "tags"]
    VIEWS_COLS = ["views", "view", "impressions"]
    REACT_COLS = ["social_reactions", "reactions", "likes"]
    SENT_COLS = ["sentiment", "sentiment_label"]
    CHAN_COLS = ["channel_id", "channel", "source"]
    LANG_COLS = ["lang", "language"]
    CONF_COLS = ["confidence", "prob", "score"]

    def find_col(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    COL_DATE = find_col(DATE_CANDIDATES)
    COL_MSG = find_col(ID_COLS)
    COL_TEXT = find_col(TEXT_COLS)
    COL_KW = find_col(KWORDS_COLS)
    COL_VW = find_col(VIEWS_COLS)
    COL_RC = find_col(REACT_COLS)
    COL_SNT = find_col(SENT_COLS)
    COL_CH = find_col(CHAN_COLS)
    COL_LNG = find_col(LANG_COLS)
    COL_CONF = find_col(CONF_COLS)

    with st.expander("Найденные столбцы", expanded=False):
        st.write({
            "date": COL_DATE, "message_id": COL_MSG, "text": COL_TEXT, "k_words": COL_KW,
            "views": COL_VW, "social_reactions": COL_RC, "sentiment": COL_SNT,
            "channel (raw)": COL_CH, "lang": COL_LNG, "confidence": COL_CONF
        })

    if not COL_DATE:
        st.error("Не найдена временная колонка (ожидались: data_utc / date_utc / date / timestamp / time).")
        st.stop()

    # Типы/преобразования
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    df = df.dropna(subset=[COL_DATE]).copy()
    df["date_day"] = df[COL_DATE].dt.normalize()

    if COL_VW and df[COL_VW].dtype == object:
        df[COL_VW] = pd.to_numeric(df[COL_VW], errors="coerce")

    if COL_RC and df[COL_RC].dtype == object:
        df[COL_RC] = pd.to_numeric(df[COL_RC], errors="coerce")

    # Человекочитаемое имя канала
    COL_CH_NAME = None
    if COL_CH:
        ch_as_str = df[COL_CH].astype(str)
        df["channel_name"] = ch_as_str.map(CHANNELS_STR).fillna(ch_as_str)
        COL_CH_NAME = "channel_name"

    # ----------------------------
    # Минимальный UI (дата + topN)
    # ----------------------------
    st.sidebar.header("Отбор")
    dmin, dmax = df["date_day"].min(), df["date_day"].max()
    dr = st.sidebar.slider(
        "Диапазон дат",
        min_value=dmin.to_pydatetime(),
        max_value=dmax.to_pydatetime(),
        value=(dmin.to_pydatetime(), dmax.to_pydatetime())
    )
    mask_time = (df["date_day"] >= pd.to_datetime(dr[0])) & (df["date_day"] <= pd.to_datetime(dr[1]))
    dff = df.loc[mask_time].copy()

    top_n = st.sidebar.slider("TOP N (посты/каналы/ключевые слова)", 5, 100, 20)

    st.markdown("---")

    # ----------------------------
    # A. KPI
    # ----------------------------
    st.subheader("A. Обзор (KPI)")
    k1, k2, k3, k4 = st.columns(4)
    total_posts = len(dff)
    with k1:
        st.metric("Всего постов", f"{total_posts:,}".replace(",", " "))
    if COL_VW:
        sum_views = int(dff[COL_VW].fillna(0).sum())
        avg_views = dff[COL_VW].fillna(0).mean()
        with k2:
            st.metric("Суммарные просмотры", f"{sum_views:,}".replace(",", " "))
        with k3:
            st.metric("Средние просмотры/пост", f"{int(avg_views):,}".replace(",", " "))
    if COL_SNT:
        cnt = dff[COL_SNT].value_counts(dropna=True)
        pos = int(cnt.get("positive", 0))
        neg = int(cnt.get("negative", 0))
        tot = int(cnt.sum()) if int(cnt.sum()) > 0 else 1
        with k4:
            st.metric("Доли настроений", f"+{pos/tot*100:.1f}% / -{neg/tot*100:.1f}%")

    st.markdown("---")

    # ----------------------------
    # B. Активность и охват
    # ----------------------------
    st.subheader("B. Активность каналов")

    # Посты по дням
    cnt_daily = dff.groupby("date_day", as_index=False).size().rename(columns={"size": "posts"})
    if not cnt_daily.empty:
        chart_posts = alt.Chart(cnt_daily).mark_bar().encode(
            x=alt.X("date_day:T", title="Дата"),
            y=alt.Y("posts:Q", title="Количество постов"),
            tooltip=["date_day:T", "posts:Q"]
        ).properties(height=250)
        st.altair_chart(chart_posts, use_container_width=True)
    else:
        st.info("Нет данных для «Постов по дням».")

    # Просмотры по дням
    if COL_VW:
        views_daily = dff.groupby("date_day", as_index=False)[COL_VW].sum()
        chart_views = alt.Chart(views_daily).mark_line().encode(
            x=alt.X("date_day:T", title="Дата"),
            y=alt.Y(f"{COL_VW}:Q", title="Сумма просмотров"),
            tooltip=["date_day:T", f"{COL_VW}:Q"]
        ).properties(height=250)
        st.altair_chart(chart_views, use_container_width=True)

    # Стек по sentiment
    if COL_SNT:
        s_daily = (dff.groupby(["date_day", COL_SNT], as_index=False)
                   .size().rename(columns={"size": "posts"}))
        chart_sent = alt.Chart(s_daily).mark_area().encode(
            x=alt.X("date_day:T", title="Дата"),
            y=alt.Y("posts:Q", stack="zero", title="Постов"),
            color=alt.Color(f"{COL_SNT}:N", title="sentiment")
        ).properties(height=250)
        st.altair_chart(chart_sent, use_container_width=True)

    st.markdown("---")

    # ----------------------------
    # C. Вовлечённость и каналы
    # ----------------------------
    st.subheader("C. Вовлечённость и каналы")
    c1, c2 = st.columns(2)

    CH_KEY = COL_CH_NAME if COL_CH_NAME else COL_CH

    if CH_KEY and COL_VW:
        with c1:
            top_channels = (dff.groupby(CH_KEY, as_index=False)[COL_VW]
                              .sum().sort_values(COL_VW, ascending=False).head(top_n))
            ch_bar = alt.Chart(top_channels).mark_bar().encode(
                x=alt.X(f"{COL_VW}:Q", title="SUM(views)"),
                y=alt.Y(f"{CH_KEY}:N", sort='-x', title="Канал"),
                tooltip=[f"{CH_KEY}:N", f"{COL_VW}:Q"]
            ).properties(height=320)
            st.altair_chart(ch_bar, use_container_width=True)

    if CH_KEY and COL_VW and COL_RC:
        with c2:
            agg = dff.groupby(CH_KEY, as_index=False)[[COL_VW, COL_RC]].sum()
            agg["ER"] = agg[COL_RC] / agg[COL_VW].replace(0, np.nan)
            agg = agg.dropna(subset=["ER"]).sort_values("ER", ascending=False).head(top_n)
            er_bar = alt.Chart(agg).mark_bar().encode(
                x=alt.X("ER:Q", title="reactions / views"),
                y=alt.Y(f"{CH_KEY}:N", sort='-x', title="Канал"),
                tooltip=[f"{CH_KEY}:N", "ER:Q"]
            ).properties(height=320)
            st.altair_chart(er_bar, use_container_width=True)

    st.markdown("---")

    # ----------------------------
    # D. ТОП посты и текст
    # ----------------------------
    st.subheader("D. ТОП посты и текст")

    if COL_VW:
        top_posts = dff.copy()
        if COL_TEXT:
            def short(s, n=140):
                s = str(s) if not pd.isna(s) else ""
                return s if len(s) <= n else s[:n-1] + "…"
            top_posts["snippet"] = top_posts[COL_TEXT].apply(short)

        if CH_KEY and CH_KEY not in top_posts.columns and COL_CH in top_posts.columns:
            ch_as_str = top_posts[COL_CH].astype(str)
            top_posts["channel_name"] = ch_as_str.map(CHANNELS_STR).fillna(ch_as_str)

        order_cols = ["date_day"]
        if CH_KEY: order_cols.append(CH_KEY)
        order_cols.append(COL_VW)
        if "snippet" in top_posts.columns: order_cols.append("snippet")

        top_viewed = (top_posts.sort_values(COL_VW, ascending=False)
                                [order_cols].head(top_n))
        st.dataframe(top_viewed, use_container_width=True)

    # ----------------------------
    # Ключевые слова + Облако слов
    # ----------------------------
    st.subheader("Ключевые слова")
    tokens = []
    if COL_KW:
        kk = dff[[COL_KW]].dropna()
        kk = kk.assign(_kw=kk[COL_KW].astype(str).str.split(","))
        kk = kk.explode("_kw")["_kw"].dropna().astype(str).str.strip().str.lower().tolist()
        tokens.extend(kk)
    elif COL_TEXT:
        txt = " ".join(map(str, dff[COL_TEXT].dropna().astype(str).tolist()))
        kk = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]{3,}", txt.lower())
        tokens.extend(kk)

    stop = set([
        "и","в","на","по","за","для","из","к","от","с","а","но","или","да","это","что","как",
        "—","-","/","\\","&","|","the","and","for","with","you","are",
        "путина"  # пример стоп-слова, подстрой по данным
    ])
    tokens = [t for t in tokens if t not in stop and len(t) > 2]

    if tokens:
        srs = pd.Series(tokens)
        top_kw = srs.value_counts().reset_index()
        top_kw.columns = ["keyword", "count"]
        top_kw_display = top_kw.head(top_n)

        kw_bar = alt.Chart(top_kw_display).mark_bar().encode(
            x=alt.X("count:Q", title="Количество"),
            y=alt.Y("keyword:N", sort='-x', title="Ключевое слово"),
            tooltip=["keyword:N", "count:Q"]
        ).properties(height=400)
        st.altair_chart(kw_bar, use_container_width=True)

        # Облако слов из полного словаря (а не только top_n)
        freq = dict(zip(top_kw["keyword"], top_kw["count"]))
        render_wordcloud_from_freq(freq, title="Облако слов по выборке")
    else:
        st.info("Не удалось извлечь ключевые слова.")

    # ----------------------------
    # Ко-встречаемость тем (heatmap)
    # ----------------------------
    st.subheader("Ко-встречаемость ключевых слов")
    if COL_KW:
        pairs = []
        rows = dff[[COL_KW]].dropna().astype(str)
        for row in rows[COL_KW].tolist():
            kws = [t.strip().lower() for t in str(row).split(",") if t.strip()]
            kws = [t for t in kws if len(t) > 2]
            seen = sorted(set(kws))
            for i in range(len(seen)):
                for j in range(i + 1, len(seen)):
                    pairs.append((seen[i], seen[j]))
        if pairs:
            cnt = Counter(pairs)
            df_heat = pd.DataFrame([{"kw1": a, "kw2": b, "count": c} for (a, b), c in cnt.items()])
            base = pd.concat([df_heat["kw1"], df_heat["kw2"]]).value_counts().head(min(top_n, 30)).index.tolist()
            df_heat = df_heat[df_heat["kw1"].isin(base) & df_heat["kw2"].isin(base)]
            if not df_heat.empty:
                heat = alt.Chart(df_heat).mark_rect().encode(
                    x=alt.X("kw1:N", sort=base, title="keyword 1"),
                    y=alt.Y("kw2:N", sort=base, title="keyword 2"),
                    color=alt.Color("count:Q", title="совместные вхождения"),
                    tooltip=["kw1", "kw2", "count"]
                ).properties(height=500)
                st.altair_chart(heat, use_container_width=True)
            else:
                st.info("Недостаточно данных для тепловой карты.")
        else:
            st.info("Недостаточно пар ключевых слов для тепловой карты.")
    else:
        st.info("Нет столбца k_words — тепловая карта недоступна.")

    # ----------------------------
    # Дополнительно: распределение confidence
    # ----------------------------
    if COL_CONF:
        st.subheader("Распределение confidence")
        conf = pd.to_numeric(dff[COL_CONF], errors="coerce").dropna()
        if not conf.empty:
            conf_df = pd.DataFrame({COL_CONF: conf})
            hist = alt.Chart(conf_df).mark_bar().encode(
                x=alt.X(f"{COL_CONF}:Q", bin=alt.Bin(maxbins=30), title="confidence"),
                y=alt.Y("count():Q", title="Количество")
            ).properties(height=300)
            st.altair_chart(hist, use_container_width=True)
