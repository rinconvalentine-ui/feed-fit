import time
import uuid
import numpy as np
import streamlit as st
from PIL import Image

# Supabase
from supabase import create_client

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None


# =========================
# Supabase Storage helpers
# =========================
def sb_client():
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        return None
    return create_client(url, key)

BUCKET = st.secrets.get("SUPABASE_BUCKET", "photos")

def sb_upload(file, folder: str) -> str:
    """
    Upload un fichier dans Supabase Storage.
    folder: "refs" ou "cands"
    Retourne le path stocké (ex: refs/170..._abcd12.jpg)
    """
    client = sb_client()
    if client is None:
        raise RuntimeError("Supabase non configuré (secrets manquants).")

    name = getattr(file, "name", "image.jpg")
    ext = name.split(".")[-1].lower() if "." in name else "jpg"
    unique = f"{folder}/{int(time.time())}_{uuid.uuid4().hex[:8]}.{ext}"
    data = file.getvalue()  # bytes

    client.storage.from_(BUCKET).upload(
        path=unique,
        file=data,
        file_options={"content-type": getattr(file, "type", None) or "image/jpeg"},
    )
    return unique

def sb_list(folder: str):
    """Liste les items dans un dossier Supabase (refs/cands)."""
    client = sb_client()
    if client is None:
        raise RuntimeError("Supabase non configuré (secrets manquants).")
    return client.storage.from_(BUCKET).list(path=folder) or []

def sb_delete(path: str):
    client = sb_client()
    if client is None:
        raise RuntimeError("Supabase non configuré (secrets manquants).")
    client.storage.from_(BUCKET).remove([path])

def sb_signed_url(path: str, seconds: int = 3600) -> str:
    """Bucket privé recommandé: on affiche via URL signée."""
    client = sb_client()
    if client is None:
        raise RuntimeError("Supabase non configuré (secrets manquants).")
    res = client.storage.from_(BUCKET).create_signed_url(path, seconds)
    return res.get("signedURL") or ""


# =========================
# Feed Fit (image scoring)
# =========================
def rgb_to_hsv_np(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    hue = np.zeros_like(cmax)
    mask = delta > 1e-6

    idx = (cmax == r) & mask
    hue[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6
    idx = (cmax == g) & mask
    hue[idx] = ((b[idx] - r[idx]) / delta[idx]) + 2
    idx = (cmax == b) & mask
    hue[idx] = ((r[idx] - g[idx]) / delta[idx]) + 4
    hue = hue / 6.0

    sat = np.zeros_like(cmax)
    sat[cmax > 1e-6] = delta[cmax > 1e-6] / cmax[cmax > 1e-6]
    val = cmax
    return np.stack([hue, sat, val], axis=-1)

def dominant_palette(rgb: np.ndarray, k: int = 5):
    pixels = rgb.reshape(-1, 3)
    if pixels.shape[0] > 60000:
        idx = np.random.choice(pixels.shape[0], 60000, replace=False)
        pixels = pixels[idx]

    if KMeans is None:
        mean = pixels.mean(axis=0).astype(int).tolist()
        return [mean] * k

    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    km.fit(pixels)
    centers = np.clip(km.cluster_centers_, 0, 255).astype(int)
    return centers.tolist()

def features(rgb: np.ndarray):
    hsv = rgb_to_hsv_np(rgb)
    sat = float(np.mean(hsv[..., 1]))
    val = float(np.mean(hsv[..., 2]))
    contrast = float(np.std(hsv[..., 2]))

    rgb_f = rgb.astype(np.float32) / 255.0
    r = float(np.mean(rgb_f[..., 0]))
    b = float(np.mean(rgb_f[..., 2]))
    warmth = float((r - b + 1.0) / 2.0)

    return {"warmth": warmth, "brightness": val, "saturation": sat, "contrast": contrast}

def build_profile(ref_rgbs):
    feats = [features(img) for img in ref_rgbs]
    pal_all = []
    for img in ref_rgbs:
        pal_all.extend(dominant_palette(img, k=5))
    pal_mean = np.mean(np.array(pal_all, dtype=np.float32), axis=0).tolist()

    return {
        "target": {
            "warmth": float(np.mean([f["warmth"] for f in feats])),
            "brightness": float(np.mean([f["brightness"] for f in feats])),
            "saturation": float(np.mean([f["saturation"] for f in feats])),
            "contrast": float(np.mean([f["contrast"] for f in feats])),
            "palette_mean_rgb": pal_mean,
        }
    }

def palette_distance(profile, cand_rgb: np.ndarray) -> float:
    cand_mean = cand_rgb.reshape(-1, 3).mean(axis=0).astype(np.float32)
    ref_mean = np.array(profile["target"]["palette_mean_rgb"], dtype=np.float32)
    return float(np.linalg.norm(cand_mean - ref_mean)) / 441.0

def score(profile, cand_rgb: np.ndarray):
    t = profile["target"]
    f = features(cand_rgb)

    dw = abs(f["warmth"] - t["warmth"])
    db = abs(f["brightness"] - t["brightness"])
    ds = abs(f["saturation"] - t["saturation"])
    dc = abs(f["contrast"] - t["contrast"])
    pd = palette_distance(profile, cand_rgb)

    diff = (0.30 * dw + 0.25 * db + 0.20 * ds + 0.15 * dc + 0.10 * pd)
    sc = int(max(0, min(100, round((1.0 - diff) * 100))))
    label = "POST" if sc >= 80 else ("MAYBE" if sc >= 65 else "NO")

    tips = []
    if f["warmth"] < t["warmth"] - 0.04:
        tips.append("Trop froid → réchauffer la balance des blancs (température +).")
    if f["saturation"] > t["saturation"] + 0.05:
        tips.append("Trop saturé → baisser saturation (−5 à −12).")
    if f["brightness"] < t["brightness"] - 0.05:
        tips.append("Trop sombre → augmenter exposition / hautes lumières.")
    if f["contrast"] > t["contrast"] + 0.03:
        tips.append("Contraste dur → adoucir contrastes / noirs.")
    if not tips:
        tips.append("Très cohérent → micro-ajustements seulement.")

    return {"score": sc, "label": label, "tips": tips, "palette": dominant_palette(cand_rgb, 5)}

def pil_to_np(pil: Image.Image, max_size=1200) -> np.ndarray:
    pil = pil.convert("RGB")
    w, h = pil.size
    scale = min(1.0, max_size / max(w, h))
    if scale < 1.0:
        pil = pil.resize((int(w * scale), int(h * scale)))
    return np.array(pil, dtype=np.uint8)


# =========================
# UI
# =========================
st.set_page_config(page_title="Feed Fit", layout="wide")
tab_feed, tab_perf = st.tabs(["Feed Fit", "Performance"])

# -------------------------
# TAB: Feed Fit
# -------------------------
with tab_feed:
    st.title("Feed Fit — IA (photos) + Bibliothèque (stockage)")

    # Check Supabase config
    if sb_client() is None:
        st.error("Supabase n’est pas configuré. Ajoute SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY dans .streamlit/secrets.toml")
        st.stop()

    st.markdown("""
    **1)** Uploade et **enregistre** tes références et candidates (bibliothèque).  
    **2)** Clique **Analyser depuis la bibliothèque** → résultats + **grille 3×3** + détails.  
    **3)** Tu peux **supprimer** n’importe quelle photo quand tu veux.
    """)

    st.divider()

    # Upload + Save to library
    up1, up2 = st.columns(2)

    with up1:
        st.subheader("📌 Références (feed)")
        ref_upload = st.file_uploader("Ajoute des références à stocker", type=["jpg","jpeg","png"], accept_multiple_files=True, key="ref_save")
        if st.button("Enregistrer ces références", disabled=not ref_upload, key="btn_save_refs"):
            for f in ref_upload:
                sb_upload(f, "refs")
            st.success("Références enregistrées ✅")
            st.rerun()

    with up2:
        st.subheader("📌 Candidates (à trier)")
        cand_upload = st.file_uploader("Ajoute des candidates à stocker", type=["jpg","jpeg","png"], accept_multiple_files=True, key="cand_save")
        if st.button("Enregistrer ces candidates", disabled=not cand_upload, key="btn_save_cands"):
            for f in cand_upload:
                sb_upload(f, "cands")
            st.success("Candidates enregistrées ✅")
            st.rerun()

    st.divider()

    # Library display + delete
    libA, libB = st.columns(2)

    with libA:
        st.subheader("📚 Bibliothèque — Références")
        refs = sb_list("refs")
        if not refs:
            st.info("Aucune référence enregistrée.")
        else:
            for it in refs[::-1]:
                path = f"refs/{it['name']}"
                url = sb_signed_url(path, 3600)
                c1, c2 = st.columns([4,1])
                with c1:
                    st.image(url, use_container_width=True)
                    st.caption(path)
                with c2:
                    if st.button("🗑️", key=f"del_ref_{path}"):
                        sb_delete(path)
                        st.rerun()

    with libB:
        st.subheader("📚 Bibliothèque — Candidates")
        cands = sb_list("cands")
        if not cands:
            st.info("Aucune candidate enregistrée.")
        else:
            for it in cands[::-1]:
                path = f"cands/{it['name']}"
                url = sb_signed_url(path, 3600)
                c1, c2 = st.columns([4,1])
                with c1:
                    st.image(url, use_container_width=True)
                    st.caption(path)
                with c2:
                    if st.button("🗑️", key=f"del_cand_{path}"):
                        sb_delete(path)
                        st.rerun()

    st.divider()

    # Analysis from library
    st.subheader("🔎 Analyse")
    analyze = st.button("Analyser depuis la bibliothèque (refs + cands)", key="btn_analyze")

    if analyze:
        refs = sb_list("refs")
        cands = sb_list("cands")

        if len(refs) < 5:
            st.error("Ajoute au moins 5 références dans la bibliothèque (idéal 10–30).")
            st.stop()
        if len(cands) < 1:
            st.error("Ajoute des candidates dans la bibliothèque.")
            st.stop()

        # Load refs -> profile
        ref_rgbs = []
        for it in refs:
            path = f"refs/{it['name']}"
            url = sb_signed_url(path, 3600)
            # Streamlit peut ouvrir l'image via URL
            pil = Image.open(requests_get_image(url)).convert("RGB")  # replaced below
            ref_rgbs.append(pil_to_np(pil))

        profile = build_profile(ref_rgbs)

        # Score candidates
        results = []
        for it in cands:
            path = f"cands/{it['name']}"
            url = sb_signed_url(path, 3600)
            pil = Image.open(requests_get_image(url)).convert("RGB")  # replaced below
            rgb = pil_to_np(pil)
            s = score(profile, rgb)
            results.append({"path": path, "url": url, "pil": pil, "filename": it["name"], **s})

        results.sort(key=lambda x: x["score"], reverse=True)

        st.success("Analyse terminée ✅")

        st.subheader("Grille Instagram (preview 3×3)")
        top9 = results[:9]
        cols = st.columns(3)
        for i, r in enumerate(top9):
            with cols[i % 3]:
                st.image(r["url"], use_container_width=True)
                st.caption(f"{r['label']} — {r['score']}/100 • {r['filename']}")

        st.subheader("Détails")
        for r in results:
            with st.expander(f"{r['label']} — {r['score']}/100 — {r['filename']}"):
                st.image(r["url"], use_container_width=True)
                for tip in r["tips"]:
                    st.write("•", tip)
                st.markdown("**Actions rapides :**")
                st.write("• Si NO → ne pas poster (ou retoucher selon tips).")
                st.write("• Si MAYBE → retouche légère + compare avec les meilleurs.")
                st.write("• Si POST → prioritaire pour le feed.")

# -------------------------
# TAB: Performance
# -------------------------
with tab_perf:
    st.title("Performance — Analyse Instagram (sans API)")

    st.markdown("""
    Tu entres les chiffres que tu vois déjà sur Instagram (moyennes 7–14 jours)  
    → l’app te donne un diagnostic + recommandations concrètes.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        avg_likes = st.number_input("Likes moyens par post", min_value=0, value=120)
        avg_comments = st.number_input("Commentaires moyens", min_value=0, value=6)
    with col2:
        avg_reach = st.number_input("Reach moyen par post", min_value=0, value=1800)
        avg_story = st.number_input("Vues moyennes des stories", min_value=0, value=450)
    with col3:
        posts_week = st.number_input("Posts / semaine", min_value=0, value=2)

    st.divider()
    engagement_rate = (avg_likes + avg_comments) / max(avg_reach, 1) * 100

    st.subheader("Diagnostic")
    st.write(f"📊 **Taux d’engagement estimé : {engagement_rate:.2f}%**")

    if engagement_rate >= 8:
        st.success("Très bon engagement → répète ce qui marche + reste cohérente.")
    elif engagement_rate >= 5:
        st.warning("Engagement correct → optimise timing + contenu + régularité.")
    else:
        st.error("Engagement faible → il faut ajuster format / heure / hooks (même sur photo).")

    st.subheader("Recommandations (action)")
    recs = []
    if avg_story < avg_reach * 0.2:
        recs.append("Stories faibles → poste 3–5 slides/jour + 1 interaction (sondage/question) vers 19–21h.")
    else:
        recs.append("Stories OK → garde 3–5 slides/jour, évite les longues stories inutiles.")

    if posts_week < 2:
        recs.append("Monte à 2 posts/semaine minimum (tu as déjà dit 2/sem, garde ça).")
    else:
        recs.append("Rythme OK → garde 2 posts/sem, améliore la qualité + cohérence.")

    recs.append("Test heures : 8–9h et 19–21h pendant 2 semaines, garde celles qui gagnent.")
    recs.append("Le feed doit rester lumineux/chaud/contraste doux (quiet luxury) → c’est ton avantage.")

    for r in recs:
        st.write("•", r)

    st.subheader("Plan 7 jours (simple)")
    st.write("• 2 posts (1 hero + 1 lifestyle)")
    st.write("• Stories quotidiennes : 3–5 slides")
    st.write("• 1 interaction/jour (sondage, question, slider)")


# =========================
# Helper: fetch image bytes from signed URL
# =========================
# On met ce helper tout en bas pour éviter d'ajouter une dépendance lourde.
import requests
from io import BytesIO

@st.cache_data(show_spinner=False, ttl=600)
def requests_get_image(url: str):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return BytesIO(r.content)
