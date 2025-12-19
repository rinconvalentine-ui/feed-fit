import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from io import BytesIO

# Optionnel (pour clustering couleurs). L'app marche même sans.
try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None

st.set_page_config(page_title="Feed Fit", layout="wide")
st.title("Feed Fit — IA (photos) + Stats + Abonnés")

tabs = st.tabs(["📸 Feed Fit (photos)", "📈 Performance", "👥 Abonnés (Excel)"])


# -----------------------------
# Utils image
# -----------------------------
def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))

def resize_for_speed(img: Image.Image, max_size=640) -> Image.Image:
    w, h = img.size
    scale = min(max_size / max(w, h), 1.0)
    if scale < 1.0:
        return img.resize((int(w * scale), int(h * scale)))
    return img

def image_stats(img_np: np.ndarray):
    # img_np: HxWx3 uint8
    arr = img_np.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    # Luminance approx
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    mean_lum = float(lum.mean())
    std_lum = float(lum.std())

    # Saturation (approx via HSV conversion cheap)
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = np.where(mx == 0, 0, (mx - mn) / mx)
    mean_sat = float(np.nanmean(sat))

    return mean_lum, std_lum, mean_sat

def dominant_palette_score(img_np: np.ndarray, k=5):
    """
    Retourne (warmth_score, palette_compactness) dans [0,1]
    - warmth: proportion de pixels "chauds" (R>G>B en moyenne)
    - compactness: si couleurs pas trop dispersées
    """
    arr = img_np.reshape(-1, 3).astype(np.float32)
    # sample pour vitesse
    if arr.shape[0] > 40000:
        idx = np.random.choice(arr.shape[0], 40000, replace=False)
        arr = arr[idx]

    # warmth simple
    r = arr[:, 0].mean()
    g = arr[:, 1].mean()
    b = arr[:, 2].mean()
    warmth = np.clip((r - b) / 255.0 + 0.5, 0, 1)  # 0..1

    if KMeans is None:
        # fallback: compactness via variance RGB
        var = arr.var(axis=0).mean()
        compact = float(np.clip(1.0 - (var / (255.0**2)), 0, 1))
        return float(warmth), compact

    km = KMeans(n_clusters=min(k, len(arr)), n_init="auto", random_state=0)
    km.fit(arr)
    centers = km.cluster_centers_
    # dispersion des centres
    center_var = centers.var(axis=0).mean()
    compact = float(np.clip(1.0 - (center_var / (255.0**2)), 0, 1))
    return float(warmth), compact

def score_photo(img: Image.Image):
    """
    Score /100 orienté "feed lumineux/chaud/cohérent"
    + conseils d'édition.
    """
    img_small = resize_for_speed(img, 640)
    np_img = pil_to_np(img_small)

    mean_lum, std_lum, mean_sat = image_stats(np_img)
    warmth, compact = dominant_palette_score(np_img)

    # Cibles approximatives pour ton style (lumineux + chaud + pas trop saturé)
    # mean_lum target ~ 0.65
    lum_score = 1.0 - min(abs(mean_lum - 0.65) / 0.35, 1.0)
    # trop de contraste dur si std_lum trop haut
    contrast_score = 1.0 - min(max(std_lum - 0.23, 0) / 0.25, 1.0)
    # saturation idéale ~ 0.22
    sat_score = 1.0 - min(abs(mean_sat - 0.22) / 0.22, 1.0)

    # pondération
    total = (
        0.35 * lum_score +
        0.20 * contrast_score +
        0.20 * sat_score +
        0.15 * warmth +
        0.10 * compact
    )
    score100 = int(round(total * 100))

    tips = []
    # conseils simples
    if mean_sat > 0.30:
        tips.append("Trop saturé → baisser saturation (-5 à -12).")
    if mean_sat < 0.14:
        tips.append("Un peu terne → + légère saturation (+3 à +8).")

    if mean_lum < 0.55:
        tips.append("Trop sombre → augmenter exposition / hautes lumières.")
    if mean_lum > 0.78:
        tips.append("Trop clair → baisser hautes lumières / blancs.")

    if std_lum > 0.30:
        tips.append("Contraste dur → adoucir contraste / noirs.")
    if warmth < 0.45:
        tips.append("Trop froid → réchauffer la température (+3 à +10).")

    if not tips:
        tips.append("Très cohérent → micro-ajustements uniquement (tons + uniformité).")

    details = {
        "luminosité_moy": round(mean_lum, 3),
        "contraste_std": round(std_lum, 3),
        "saturation_moy": round(mean_sat, 3),
        "warmth": round(warmth, 3),
        "cohérence_palette": round(compact, 3),
    }

    return score100, tips, details


# =========================
# TAB 1 — PHOTOS
# =========================
with tabs[0]:
    st.subheader("📸 Feed Fit (photos) — Upload + Top 9 + conseils")

    st.markdown("**1) Upload 10–30 photos** → l’app calcule un score /100 et te sort un **Top 9**.")
    photos = st.file_uploader(
        "📷 Uploade tes photos (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if not photos:
        st.info("⬆️ Ajoute des photos pour lancer l’analyse.")
    else:
        # Analyse
        results = []
        for f in photos:
            try:
                img = Image.open(f)
                score, tips, details = score_photo(img)
                # on garde une miniature pour affichage
                thumb = resize_for_speed(img, 420)
                results.append({
                    "filename": f.name,
                    "score": score,
                    "tips": tips,
                    "details": details,
                    "thumb": thumb
                })
            except Exception as e:
                results.append({
                    "filename": f.name,
                    "score": -1,
                    "tips": [f"Erreur lecture image: {e}"],
                    "details": {},
                    "thumb": None
                })

        # Tri
        ok_results = [r for r in results if r["score"] >= 0]
        ok_results.sort(key=lambda x: x["score"], reverse=True)

        st.divider()
        st.header("Résultats")

        # TOP 9
        st.subheader("Top 9 (pré-sélection)")
        top9 = ok_results[:9]

        if not top9:
            st.error("Aucune image n’a pu être analysée.")
        else:
            cols = st.columns(3)
            for i, r in enumerate(top9):
                with cols[i % 3]:
                    st.image(r["thumb"], use_container_width=True)
                    st.markdown(f"**POST — {r['score']}/100**")
                    st.caption(r["filename"])

            # liste texte comme avant
            st.markdown("### Liste (Top 9)")
            for r in top9:
                st.write(f"**POST — {r['score']}/100**  \n{r['filename']}")

        # Détails (avec miniatures)
        st.subheader("Détails")
        for r in top9:
            with st.expander(f"POST — {r['score']}/100 — {r['filename']}"):
                st.image(r["thumb"], use_container_width=True)
                for t in r["tips"]:
                    st.write(f"• {t}")
                if r["details"]:
                    st.caption(f"Détails: {r['details']}")

        # Preview général (toutes les images uploadées)
        st.subheader("Aperçu de toutes les photos uploadées")
        grid = st.columns(4)
        for i, r in enumerate(ok_results):
            with grid[i % 4]:
                if r["thumb"] is not None:
                    st.image(r["thumb"], use_container_width=True)
                    st.caption(f"{r['filename']} — {r['score']}/100")


# =========================
# TAB 2 — PERFORMANCE
# =========================
with tabs[1]:
    st.subheader("📈 Performance Instagram")

    likes = st.number_input("Likes moyens", min_value=0, value=120, step=1)
    coms = st.number_input("Commentaires moyens", min_value=0, value=6, step=1)
    reach = st.number_input("Reach moyen", min_value=0, value=1800, step=10)

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


# =========================
# TAB 3 — ABONNÉS (EXCEL)
# =========================
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
