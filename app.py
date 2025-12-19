import streamlit as st
import pandas as pd

st.set_page_config(page_title="Feed Fit", layout="wide")

# ============== UI ==============
st.title("Feed Fit — IA + Stats + Abonnés")

tabs = st.tabs(["📸 Feed Fit (photos)", "📈 Performance", "👥 Abonnés (Excel)"])

# ============== TAB 1 ==============
with tabs[0]:
    st.subheader("📸 Feed Fit (photos)")
    st.info("Ici tu mets/laisseras ta partie analyse photos (upload + scoring).")
    st.write("✅ (On peut la recoller après si tu veux, mais là on débloque l'upload abonnés.)")

# ============== TAB 2 ==============
with tabs[1]:
    st.subheader("📈 Performance Instagram")

    likes = st.number_input("Likes moyens", min_value=0, value=120, step=1)
    coms = st.number_input("Commentaires moyens", min_value=0, value=6, step=1)
    reach = st.number_input("Reach moyen", min_value=0, value=1800, step=10)

    # petit calcul simple
    engagement = 0.0
    if reach > 0:
        engagement = ((likes + coms) / reach) * 100

    st.metric("Taux d'engagement estimé", f"{engagement:.2f}%")

    if engagement >= 6:
        st.success("Engagement correct")
    else:
        st.warning("Engagement à améliorer")

    st.markdown("**Conseils • 2 posts / semaine minimum**")
    st.markdown("- Lumière + chaleur + cohérence\n- Stories quotidiennes courtes")

# ============== TAB 3 ==============
with tabs[2]:
    st.subheader("👥 Abonnés — Comparer Followers vs Following (Excel)")

    st.markdown("""
    **Format attendu :**
    - 1 fichier **.xlsx**
    - 2 feuilles : **Followers** et **Following**
    - Dans chaque feuille : colonne A avec en A1 : `username`
    """)

    file = st.file_uploader("📂 Upload ton fichier Excel (Followers/Following)", type=["xlsx"])

    if file:
        # lecture feuilles (on accepte plusieurs variantes de noms)
        def read_sheet(xls, names):
            for n in names:
                try:
                    return pd.read_excel(xls, sheet_name=n)
                except Exception:
                    pass
            return None

        followers_df = read_sheet(file, ["Followers", "followers", "FOLLOWERS"])
        following_df = read_sheet(file, ["Following", "following", "FOLLOWING"])

        if followers_df is None or following_df is None:
            st.error("❌ Je ne trouve pas les feuilles Followers / Following dans ton Excel.")
            st.stop()

        if "username" not in followers_df.columns or "username" not in following_df.columns:
            st.error("❌ Il manque la colonne `username` (mets `username` en A1 dans chaque feuille).")
            st.stop()

        followers_df["username"] = followers_df["username"].astype(str).str.lower().str.strip()
        following_df["username"] = following_df["username"].astype(str).str.lower().str.strip()

        set_followers = set(followers_df["username"])
        set_following = set(following_df["username"])

        non_follow_back = sorted(list(set_following - set_followers))
        mutuals = sorted(list(set_following & set_followers))

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"### ❌ Ne te suivent pas en retour ({len(non_follow_back)})")
            st.dataframe(pd.DataFrame(non_follow_back, columns=["username"]), use_container_width=True)
            st.download_button(
                "⬇️ Télécharger non_follow_back.csv",
                pd.DataFrame(non_follow_back, columns=["username"]).to_csv(index=False),
                "non_follow_back.csv",
                mime="text/csv"
            )

        with col2:
            st.write(f"### 💙 Followers réciproques ({len(mutuals)})")
            st.dataframe(pd.DataFrame(mutuals, columns=["username"]), use_container_width=True)
            st.download_button(
                "⬇️ Télécharger mutuals.csv",
                pd.DataFrame(mutuals, columns=["username"]).to_csv(index=False),
                "mutuals.csv",
                mime="text/csv"
            )
    else:
        st.info("⬆️ Uploade ton Excel ici pour voir l'analyse.")
