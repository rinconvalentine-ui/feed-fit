import time
import uuid
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

from supabase import create_client

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None


# =====================================================
# CONFIG SUPABASE
# =====================================================
def sb_client():
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        return None
    return create_client(url, key)

BUCKET = st.secrets.get("SUPABASE_BUCKET", "photos")


def sb_upload(file, folder: str):
    client = sb_client()
    if client is None:
        return None

    ext = file.name.split(".")[-1].lower() if "." in file.name else "jpg"
    name = f"{folder}/{int(time.time())}_{uuid.uuid4().hex[:8]}.{ext}"
    client.storage.from_(BUCKET).upload(
        name,
        file.getvalue(),
        file_options={"content-type": file.type or "image/jpeg"},
    )
    return name


def sb_list(folder: str):
    """
    Liste les fichiers Supabase d'un dossier logique (refs / cands)
    sans erreur même si vide.
    """
    client = sb_client()
    if client is None:
        return []

    try:
        items = client.storage.from_(BUCKET).list(path="")
    except Exception:
        return []

    fichiers = []
    prefix = f"{folder}/"
    for it in items or []:
        nom = it.get("name", "")
        if nom.startswith(prefix):
            fichiers.append({"name": nom.replace(prefix, "")})
    return fichiers


def sb_delete(path: str):
    client = sb_client()
    if client:
        client.storage.from_(BUCKET).remove([path])


def sb_signed_url(path: str, seconds=3600):
    client = sb_client()
    if client is None:
        return ""
    res = client.storage.from_(BUCKET).create_signed_url(path, seconds)
    return res.get("signedURL", "")


@st.cache_data(ttl=600)
def fetch_image(url: str):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return BytesIO(r.content)


# =====================================================
# FEED FIT — ANALYSE IMAGES
# =====================================================
def rgb_to_hsv_np(rgb):
    rgb = rgb.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    delta = cmax - cmin

    hue = np.zeros_like(cmax)
    mask = delta > 1e-6
    idx = (cmax == r) & mask
    hue[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6
    idx = (cmax == g) & mask
    hue[idx] = ((b[idx] - r[idx]) / delta[idx]) + 2
    idx = (cmax == b) & mask
    hue[idx] = ((r[idx] - g[idx]) / delta[idx]) + 4
    hue /= 6.0

    sat = np.zeros_like(cmax)
    sat[cmax > 0] = delta[cmax > 0] / cmax[cmax > 0]
    val = cmax
    return np.stack([hue, sat, val], axis=-1)


def features(rgb):
    hsv = rgb_to_hsv_np(rgb)
    return {
        "saturation": float(np.mean(hsv[..., 1])),
        "luminosite": float(np.mean(hsv[..., 2])),
        "contraste": float(np.std(hsv[..., 2])),
        "chaleur": float((np.mean(rgb[..., 0]) - np.mean(rgb[..., 2]) + 255) / 510),
    }


def build_profile(refs):
    feats = [features(img) for img in refs]
    return {
        "saturation": np.mean([f["saturation"] for f in feats]),
        "luminosite": np.mean([f["luminosite"] for f in feats]),
        "contraste": np.mean([f["contraste"] for f in feats]),
        "chaleur": np.mean([f["chaleur"] for f in feats]),
    }


def score_image(profile, rgb):
    f = features(rgb)
    diff = (
        abs(f["saturation"] - profile["saturation"]) * 0.3
        + abs(f["luminosite"] - profile["luminosite"]) * 0.3
        + abs(f["contraste"] - profile["contraste"]) * 0.2
        + abs(f["chaleur"] - profile["chaleur"]) * 0.2
    )
    score = max(0, min(100, int((1 - diff) * 100)))

    label = "POST" if score >= 80 else "MAYBE" if score >= 65 else "NO"

    conseils = []
    if f["saturation"] > profile["saturation"] + 0.05:
        conseils.append("Trop saturée → baisser saturation.")
    if f["luminosite"] < profile["luminosite"] - 0.05:
        conseils.append("Trop sombre → augmenter exposition.")
    if f["contraste"] > profile["contraste"] + 0.03:
        conseils.append("Contraste trop dur → adoucir noirs.")

    if not conseils:
        conseils.append("Très cohérente avec ton feed.")

    return score, label, conseils


def pil_to_np(pil):
    return np.array(pil.convert("RGB"), dtype=np.uint8)


# =====================================================
# UI
# =====================================================
st.set_page_config(page_title="Feed Fit", layout="wide")

if sb_client() is None:
    st.error("❌ Supabase non configuré. Ajoute les clés dans Streamlit > Manage app > Secrets.")
    st.stop()

tab_feed, tab_perf = st.tabs(["Feed Fit", "Performance"])

# ---------------- FEED FIT ----------------
with tab_feed:
    st.title("Feed Fit — IA + Bibliothèque")

    st.markdown("""
    1️⃣ Stocke tes photos (références + candidates)  
    2️⃣ Analyse depuis la bibliothèque  
    3️⃣ Garde seulement les meilleures pour ton feed
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Références")
        files = st.file_uploader("Ajouter des références", accept_multiple_files=True, type=["jpg","png","jpeg"])
        if st.button("Enregistrer références"):
            for f in files or []:
                sb_upload(f, "refs")
            st.success("Références enregistrées")
            st.rerun()

    with col2:
        st.subheader("📌 Candidates")
        files = st.file_uploader("Ajouter des candidates", accept_multiple_files=True, type=["jpg","png","jpeg"], key="cand")
        if st.button("Enregistrer candidates"):
            for f in files or []:
                sb_upload(f, "cands")
            st.success("Candidates enregistrées")
            st.rerun()

    st.divider()

    st.subheader("📚 Bibliothèque")
    left, right = st.columns(2)

    with left:
        st.markdown("### Références")
        for it in sb_list("refs"):
            path = f"refs/{it['name']}"
            url = sb_signed_url(path)
            st.image(url, use_container_width=True)
            if st.button("🗑️ Supprimer", key=path):
                sb_delete(path)
                st.rerun()

    with right:
        st.markdown("### Candidates")
        for it in sb_list("cands"):
            path = f"cands/{it['name']}"
            url = sb_signed_url(path)
            st.image(url, use_container_width=True)
            if st.button("🗑️ Supprimer", key=path):
                sb_delete(path)
                st.rerun()

    st.divider()

    if st.button("🔎 Analyser depuis la bibliothèque"):
        refs = sb_list("refs")
        cands = sb_list("cands")

        if len(refs) < 5:
            st.error("Ajoute au moins 5 références.")
            st.stop()

        ref_imgs = []
        for r in refs:
            img = Image.open(fetch_image(sb_signed_url(f"refs/{r['name']}")))
            ref_imgs.append(pil_to_np(img))

        profile = build_profile(ref_imgs)

        results = []
        for c in cands:
            img = Image.open(fetch_image(sb_signed_url(f"cands/{c['name']}")))
            rgb = pil_to_np(img)
            s, l, tips = score_image(profile, rgb)
            results.append((s, l, tips, img, c["name"]))

        results.sort(reverse=True, key=lambda x: x[0])

        st.subheader("🟩 Grille Instagram (Top 9)")
        cols = st.columns(3)
        for i, r in enumerate(results[:9]):
            with cols[i % 3]:
                st.image(r[3], use_container_width=True)
                st.caption(f"{r[1]} — {r[0]}/100")

        st.subheader("📋 Détails")
        for r in results:
            with st.expander(f"{r[1]} — {r[0]}/100 — {r[4]}"):
                st.image(r[3], use_container_width=True)
                for t in r[2]:
                    st.write("•", t)

# ---------------- PERFORMANCE ----------------
with tab_perf:
    st.title("Performance Instagram")

    likes = st.number_input("Likes moyens", 0, 10000, 120)
    comments = st.number_input("Commentaires moyens", 0, 1000, 6)
    reach = st.number_input("Reach moyen", 0, 100000, 1800)

    engagement = (likes + comments) / max(reach, 1) * 100
    st.metric("Taux d'engagement estimé", f"{engagement:.2f}%")

    if engagement >= 8:
        st.success("Excellent engagement")
    elif engagement >= 5:
        st.warning("Engagement correct")
    else:
        st.error("Engagement faible")

    st.markdown("""
    **Conseils**
    • 2 posts / semaine minimum  
    • Lumière + chaleur + cohérence  
    • Stories quotidiennes courtes
    """)
