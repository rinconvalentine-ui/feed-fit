import streamlit as st
from PIL import Image
import numpy as np

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None

st.set_page_config(page_title="Feed Fit", layout="wide")
st.title("Feed Fit — IA pour choisir tes photos (IG + TikTok)")

st.markdown("""
**Étape 1 :** upload 10–30 photos qui représentent ton feed idéal (références).  
**Étape 2 :** upload des candidates → score “ça colle au feed ?” + conseils retouche.  
✅ **Nouveau :** preview **grille Instagram 3×3** avec les images.
""")

# ---------- Utils ----------
def load_image_np(file, max_size=900):
    """Charge l'image en numpy (RGB). Resize pour accélérer."""
    img = Image.open(file).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_size / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    return np.array(img, dtype=np.uint8)

def rgb_to_hsv_np(rgb):
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

def dominant_palette(rgb, k=5):
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

def features(rgb):
    hsv = rgb_to_hsv_np(rgb)
    sat = float(np.mean(hsv[..., 1]))
    val = float(np.mean(hsv[..., 2]))
    contrast = float(np.std(hsv[..., 2]))

    rgb_f = rgb.astype(np.float32) / 255.0
    r = float(np.mean(rgb_f[..., 0]))
    b = float(np.mean(rgb_f[..., 2]))
    warmth = float((r - b + 1.0) / 2.0)

    return {"warmth": warmth, "brightness": val, "saturation": sat, "contrast": contrast}

def build_profile(ref_imgs_np):
    feats = [features(img) for img in ref_imgs_np]

    pal_all = []
    for img in ref_imgs_np:
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

def palette_distance(profile, cand_rgb):
    cand_mean = cand_rgb.reshape(-1, 3).mean(axis=0).astype(np.float32)
    ref_mean = np.array(profile["target"]["palette_mean_rgb"], dtype=np.float32)
    return float(np.linalg.norm(cand_mean - ref_mean)) / 441.0

def score(profile, cand_rgb):
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

# ---------- UI ----------
col1, col2 = st.columns(2)
with col1:
    st.subheader("1) Références feed")
    ref_files = st.file_uploader("Photos de référence", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

with col2:
    st.subheader("2) Photos candidates")
    cand_files = st.file_uploader("Photos candidates", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if "profile" not in st.session_state:
    st.session_state.profile = None

if st.button("Créer / Mettre à jour le profil feed"):
    if not ref_files or len(ref_files) < 5:
        st.error("Mets au moins 5 photos de référence (idéal : 10–30).")
    else:
        refs_np = [load_image_np(f) for f in ref_files]
        st.session_state.profile = build_profile(refs_np)
        st.success("Profil feed créé ✅")

profile = st.session_state.profile

if profile and cand_files:
    st.divider()
    st.header("Résultats")

    # Analyse candidates + garder l'image pour affichage
    results = []
    for f in cand_files:
        pil = Image.open(f).convert("RGB")   # pour affichage
        img = np.array(pil, dtype=np.uint8)  # pour scoring
        s = score(profile, img)
        results.append({"filename": f.name, "pil": pil, **s})

    results.sort(key=lambda x: x["score"], reverse=True)

    # --- Instagram grid preview 3x3
    st.subheader("Grille Instagram (preview 3×3)")
    top9 = results[:9]
    cols = st.columns(3)
    for i, r in enumerate(top9):
        with cols[i % 3]:
            st.image(r["pil"], use_container_width=True)
            st.caption(f"{r['label']} — {r['score']}/100 • {r['filename']}")

    # --- Détails
    st.subheader("Détails")
    for r in results:
        with st.expander(f"{r['label']} — {r['score']}/100 — {r['filename']}"):
            st.image(r["pil"], use_container_width=True)
            for tip in r["tips"]:
                st.write("•", tip)

else:
    st.info("Crée d’abord le profil feed, puis ajoute des candidates.")
