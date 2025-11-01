# streamlit_app.py (фрагмент)
import streamlit as st
import pandas as pd
from news_matcher import rag_match_across_channels

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

def resolve_username(ch_id):
    try:
        return CHANNELS.get(int(ch_id)) or CHANNELS.get(ch_id) or str(ch_id)
    except Exception:
        return str(ch_id)

q = st.text_input("Запрос", "ИИ новости за неделю")

if st.button("Найти дубликаты"):
    out = rag_match_across_channels(q)
    seed = out.get("seed")
    dups = out.get("duplicates", [])

    if seed:
        st.subheader("Оригинал (seed)")
        seed_user = resolve_username(seed["channel_id"])
        seed_link = seed.get("link")
        st.markdown(f"**Канал:** `{seed_user}`  •  **Сообщение:** [{seed['message_id']}]({seed_link})")
        st.write(seed["text"])

        st.subheader(f"Найденные дубликаты: {len(dups)}")
        if dups:
            rows = []
            for d in dups:
                p = d["payload"]
                ch_id = p.get("channel_id")
                uname = resolve_username(ch_id)
                link = d.get("link")
                rows.append({
                    "Канал": uname,
                    "channel_id": ch_id,
                    "message_id": p.get("message_id"),
                    "Ссылка": link,
                    "Итоговый скор": d["score_final"],
                    "Cosine": d["score_cosine"],
                    "Jaccard": d["score_jaccard"],
                    "Текст (превью)": (d["text"] or "")[:180].replace("\n"," ")
                })
            df = pd.DataFrame(rows).sort_values(["Итоговый скор","Cosine"], ascending=False)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Дубликатов не найдено.")
    else:
        st.info(out.get("message", "Ничего не найдено."))
