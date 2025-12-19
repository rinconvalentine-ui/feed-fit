import os
import io
import re
import json
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance

# sklearn (optionnel mais recommandé)
try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None

# =========================
# CONFIG & STORAGE
# =========================
st.set_page_config(page_title="Feed Fit — IA + Stats + Abonnés", layout="wide")

DATA_DIR = "data"
PHOTOS_DIR = os.path.join(DATA_DIR, "photos")
REF_DIR = os.path.join(DATA_DIR, "refs")
META_PATH = os.path.join(DATA_DIR, "photo_meta.json")
INTERACTIONS_PATH = os.path.join(DATA_DIR, "interactions.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PHOTOS_DIR, exist_ok=True)
os.makedirs(REF_DIR, exist_ok=True)

def _load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def _save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_meta():
    return _load_json(META_PATH, default={"photos": {}, "refs": {}})

def save_meta(meta):
    _save_json(META_PATH, meta)

def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def safe_filename(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^a-zA-Z0-9._-]", "", name)
    return name or "image"

def list_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".webp")
    files = []
    for fn in os.listdir(folder):
        if fn.lower().endswith(exts):
            files.append(fn)
    return sorted(files)

# =========================
# IMAGE FEATURES (simple & fiable)
# =========================
def pil_to_np(img: Image.Image, max_side=600):
    img = img.convert("RGB")
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    arr = np.asarray(img).astype(np.float32) / 255.0
    return img, arr

def rgb_to_hsv_np(arr):
    # arr: HxWx3 in [0,1]
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = np.max(arr, axis=-1)
    mn = np.min(arr, axis=-1)
    diff = mx - mn

    h = np.zeros_like(mx)
    s = np.zeros_like(mx)
    v = mx

    s[mx != 0] = diff[mx != 0] / mx[mx != 0]

    mask = diff != 0
    r_eq = (mx == r) & mask
    g_eq = (mx == g) & mask
    b_eq = (mx == b) & mask

    h[r_eq] = ((g[r_eq] - b[r_eq]) / diff[r_eq]) % 6
    h[g_eq] = ((b[g_eq] - r[g_eq]) / diff[g_eq]) + 2
    h[b_eq] = ((r[b_eq] - g[b_eq]) / diff[b_eq]) + 4
    h = h / 6.0
    return h, s, v

def compute_features(img: Image.Image):
    _, arr = pil_to_np(img)
    h, s, v = rgb_to_hsv_np(arr)

    brightness = float(np.mean(v))            # 0..1
    saturation = float(np.mean(s))            # 0..1
    contrast = float(np.std(v))               # ~0..0.3+

    # Warmth: compare red vs blue
    r = arr[..., 0]
    b = arr[..., 2]
    warmth = float(np.mean(r - b))            # negative = froid, positive = chaud

    # Beige/sable score (proche de teintes chaudes claires)
    # On approx : hue ~ [0.05..0.18], sat modérée, value élevée
    mask_beige = (h > 0.05) & (h < 0.18) & (s > 0.10) & (s < 0.45) & (v > 0.55)
    beige_ratio = float(np.mean(mask_beige))

    return {
        "brightness": brightness,
        "saturation": saturation,
        "contrast": contrast,
        "warmth": warmth,
        "beige_ratio": beige_ratio
    }

def score_to_target(feat, target):
    # target dict with desired ranges / weights
    # convert to 0..100
    def closeness(x, mu, tol):
        # 1 at mu, 0 at mu +/- tol or more
        d = abs(x - mu)
        return max(0.0, 1.0 - d / tol)

    b = closeness(feat["brightness"], target["b_mu"], target["b_tol"])
    s = closeness(feat["saturation"], target["s_mu"], target["s_tol"])
    c = closeness(feat["contrast"], target["c_mu"], target["c_tol"])
    w = closeness(feat["warmth"], target["w_mu"], target["w_tol"])
    z = closeness(feat["beige_ratio"], target["z_mu"], target["z_tol"])

    score = (
        target["wb"] * b +
        target["ws"] * s +
        target["wc"] * c +
        target["ww"] * w +
        target["wz"] * z
    ) / (target["wb"] + target["ws"] + target["wc"] + target["ww"] + target["wz"] + 1e-9)

    return int(round(100 * score))

def advice_from_features(feat, target):
    adv = []

    if feat["saturation"] > target["s_mu"] + target["s_tol"]*0.35:
        adv.append("Trop saturé → baisse saturation (≈ -5 à -15).")
    if feat["saturation"] < target["s_mu"] - target["s_tol"]*0.35:
        adv.append("Trop terne → augmente légèrement saturation (+3 à +8).")

    if feat["brightness"] < target["b_mu"] - target["b_tol"]*0.35:
        adv.append("Trop sombre → augmente exposition / hautes lumières (+).")
    if feat["brightness"] > target["b_mu"] + target["b_tol"]*0.35:
        adv.append("Trop clair → baisse hautes lumières / blancs (-).")

    if feat["contrast"] > target["c_mu"] + target["c_tol"]*0.35:
        adv.append("Contraste dur → adoucir noirs / contraste (-).")
    if feat["contrast"] < target["c_mu"] - target["c_tol"]*0.35:
        adv.append("Manque de contraste → ajoute un peu de contraste (+).")

    if feat["warmth"] < target["w_mu"] - target["w_tol"]*0.35:
        adv.append("Trop froid → réchauffe la balance des blancs (+temp).")
    if feat["warmth"] > target["w_mu"] + target["w_tol"]*0.35:
        adv.append("Trop chaud/orangé → refroidis légèrement (-temp).")

    if feat["beige_ratio"] < target["z_mu"] - target["z_tol"]*0.35:
        adv.append("Pas assez sable/beige → privilégie tons chauds clairs (sable, crème, sunset).")

    if not adv:
        adv.append("Cohérent avec ta cible (lumière/chaud/beige).")

    return adv

def make_thumb(img: Image.Image, max_side=420):
    img = img.convert("RGB")
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    return img

# =========================
# INTERACTIONS (posts/likes)
# =========================
def load_interactions():
    if os.path.exists(INTERACTIONS_PATH):
        try:
            return pd.read_csv(INTERACTIONS_PATH)
        except Exception:
            pass
    return pd.DataFrame(columns=["post_url", "username", "event", "created_at"])

def save_interactions(df):
    df.to_csv(INTERACTIONS_PATH, index=False)

def parse_usernames(text: str):
    if not text:
        return []
    cleaned = text.replace("\n", ",").replace(";", ",").replace(" ", ",")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    out = []
    for p in parts:
        p = p.lstrip("@").strip()
        p = re.sub(r"[^a-zA-Z0-9._]", "", p)
        if p:
            out.append(p.lower())
    return sorted(set(out))

# =========================
# EXCEL SHEETS (followers/following)
# =========================
def normalize_sheet_name(x):
    return re.sub(r"\s+", "", str(x).strip().lower())

def find_sheet(df_dict, wanted):
    want = normalize_sheet_name(wanted)
    for k in df_dict.keys():
        if normalize_sheet_name(k) == want:
            return k
    return None

def extract_usernames_from_sheet(df):
    # accepte colonne A sans header, ou header "username"
    if df is None or df.empty:
        return []
    cols = [str(c).strip().lower() for c in df.columns]
    if "username" in cols:
        s = df[df.columns[cols.index("username")]]
    else:
        s = df.iloc[:, 0]
    s = s.astype(str).str.strip()
    s = s[s != ""]
    s = s.str.replace("@", "", regex=False)
    s = s.str.lower()
    # supprime "nan"
    s = s[s != "nan"]
    return sorted(set(s.tolist()))

# =========================
# UI
# =========================
st.title("Feed Fit — IA (photos) + Posts/Likes + Abonnés (Excel) + Performance")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📸 Feed Fit (photos)",
    "🗂️ Bibliothèque (stockage)",
    "👥 Abonnés (Excel)",
    "🧾 Posts & Likes (manuel)",
    "📈 Performance (manuel/CSV)"
])

meta = load_meta()

# -------------------------
# TAB 1: FEED FIT (PHOTOS)
# -------------------------
with tab1:
    st.subheader("Feed Fit (photos) — analyse + Top 9 + conseils + comparaison")

    st.markdown("""
**Objectif :** t’aider à choisir les photos qui collent à ton feed (lumineux, chaud, sable/beige, luxe discret)  
✅ **Tu vois chaque photo + son nom**, score /100 + conseils retouche.
""")

    # Target controls
    with st.expander("🎯 Réglage de ta cible (feed)", expanded=True):
        colA, colB, colC = st.columns(3)

        with colA:
            b_mu = st.slider("Luminosité cible", 0.35, 0.85, 0.68, 0.01)
            b_tol = st.slider("Tolérance luminosité", 0.05, 0.30, 0.16, 0.01)
            c_mu = st.slider("Contraste cible", 0.02, 0.20, 0.08, 0.005)
            c_tol = st.slider("Tolérance contraste", 0.01, 0.20, 0.06, 0.005)

        with colB:
            s_mu = st.slider("Saturation cible", 0.05, 0.70, 0.28, 0.01)
            s_tol = st.slider("Tolérance saturation", 0.05, 0.50, 0.18, 0.01)
            w_mu = st.slider("Chaleur cible (rouge>bleu)", -0.15, 0.25, 0.06, 0.01)
            w_tol = st.slider("Tolérance chaleur", 0.05, 0.40, 0.16, 0.01)

        with colC:
            z_mu = st.slider("Taux sable/beige cible", 0.00, 0.45, 0.10, 0.01)
            z_tol = st.slider("Tolérance sable/beige", 0.02, 0.50, 0.10, 0.01)

            st.caption("Pondérations (importance de chaque critère)")
            wb = st.slider("Poids luminosité", 1, 10, 7)
            ws = st.slider("Poids saturation", 1, 10, 5)
            wc = st.slider("Poids contraste", 1, 10, 4)
            ww = st.slider("Poids chaleur", 1, 10, 6)
            wz = st.slider("Poids sable/beige", 1, 10, 6)

    target = {
        "b_mu": b_mu, "b_tol": b_tol,
        "s_mu": s_mu, "s_tol": s_tol,
        "c_mu": c_mu, "c_tol": c_tol,
        "w_mu": w_mu, "w_tol": w_tol,
        "z_mu": z_mu, "z_tol": z_tol,
        "wb": wb, "ws": ws, "wc": wc, "ww": ww, "wz": wz
    }

    st.divider()

    st.write("### 1) Ajoute tes photos candidates (elles sont stockées)")
    uploaded = st.file_uploader(
        "Upload (JPG/PNG/WEBP) — tu peux en sélectionner plusieurs",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )

    if uploaded:
        for uf in uploaded:
            b = uf.getvalue()
            h = sha1_bytes(b)
            base = safe_filename(os.path.splitext(uf.name)[0])
            ext = os.path.splitext(uf.name)[1].lower() or ".jpg"
            fn = f"{base}_{h[:10]}{ext}"
            out_path = os.path.join(PHOTOS_DIR, fn)
            if not os.path.exists(out_path):
                with open(out_path, "wb") as f:
                    f.write(b)
                meta["photos"][fn] = {
                    "original_name": uf.name,
                    "sha1": h,
                    "added_at": datetime.utcnow().isoformat()
                }
        save_meta(meta)
        st.success("Photos ajoutées à ta bibliothèque ✅")

    # Optional: reference feed images (inspiration)
    st.write("### 2) Ajoute des photos de référence (ton feed idéal / inspiration)")
    ref_up = st.file_uploader(
        "Upload références (optionnel) — pour comparaison",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="ref_uploader"
    )
    if ref_up:
        for uf in ref_up:
            b = uf.getvalue()
            h = sha1_bytes(b)
            base = safe_filename(os.path.splitext(uf.name)[0])
            ext = os.path.splitext(uf.name)[1].lower() or ".jpg"
            fn = f"ref_{base}_{h[:10]}{ext}"
            out_path = os.path.join(REF_DIR, fn)
            if not os.path.exists(out_path):
                with open(out_path, "wb") as f:
                    f.write(b)
                meta["refs"][fn] = {
                    "original_name": uf.name,
                    "sha1": h,
                    "added_at": datetime.utcnow().isoformat()
                }
        save_meta(meta)
        st.success("Références ajoutées ✅")

    st.divider()

    # Load candidates
    candidates = list_images(PHOTOS_DIR)
    if not candidates:
        st.info("Ajoute des photos candidates pour lancer l’analyse.")
    else:
        # Compute reference “average” (optional)
        ref_files = list_images(REF_DIR)
        ref_feats = []
        for rf in ref_files[:30]:
            try:
                img = Image.open(os.path.join(REF_DIR, rf))
                ref_feats.append(compute_features(img))
            except Exception:
                pass

        if ref_feats:
            # show reference summary
            df_ref = pd.DataFrame(ref_feats)
            st.caption("Références (moyenne) : luminosité/saturation/contraste/chaleur/beige")
            st.write(df_ref.mean(numeric_only=True).round(3))

        st.write("### 3) Résultats (tu vois la photo + le nom + le score)")
        results = []

        # Analyse each candidate
        for fn in candidates:
            path = os.path.join(PHOTOS_DIR, fn)
            try:
                img = Image.open(path)
            except Exception:
                continue

            feat = compute_features(img)
            score = score_to_target(feat, target)
            adv = advice_from_features(feat, target)

            results.append({
                "filename": fn,
                "score": score,
                **feat,
                "advice": adv
            })

        if not results:
            st.warning("Aucune image analysable.")
        else:
            df = pd.DataFrame(results).sort_values("score", ascending=False)

            # Top 9
            st.subheader("Top 9 (pré-sélection)")
            top9 = df.head(9).copy()

            cols = st.columns(3)
            for i, row in enumerate(top9.itertuples(index=False)):
                c = cols[i % 3]
                with c:
                    img = Image.open(os.path.join(PHOTOS_DIR, row.filename))
                    st.image(make_thumb(img, 520), caption=f"{row.filename} — {row.score}/100", use_container_width=True)

            st.divider()

            # Full list with expanders
            st.subheader("Détails (chaque photo + conseils)")
            for row in df.itertuples(index=False):
                with st.expander(f"{row.score}/100 — {row.filename}", expanded=False):
                    img = Image.open(os.path.join(PHOTOS_DIR, row.filename))
                    st.image(make_thumb(img, 720), use_container_width=True)

                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.write("**Mesures**")
                        st.write({
                            "luminosité": round(row.brightness, 3),
                            "saturation": round(row.saturation, 3),
                            "contraste": round(row.contrast, 3),
                            "chaleur": round(row.warmth, 3),
                            "sable/beige": round(row.beige_ratio, 3),
                        })
                    with col2:
                        st.write("**Conseils retouche (pour ton feed)**")
                        for a in row.advice:
                            st.write("• " + a)

                    st.write("**Actions**")
                    del_col, keep_col = st.columns([1, 3])
                    with del_col:
                        if st.button("🗑️ Supprimer cette photo", key=f"del_{row.filename}"):
                            try:
                                os.remove(os.path.join(PHOTOS_DIR, row.filename))
                            except Exception:
                                pass
                            meta = load_meta()
                            meta["photos"].pop(row.filename, None)
                            save_meta(meta)
                            st.rerun()
                    with keep_col:
                        st.caption("Suppression = enlève du stockage. (Le score se recalculera automatiquement.)")


# -------------------------
# TAB 2: LIBRARY (STOCKAGE)
# -------------------------
with tab2:
    st.subheader("Bibliothèque — stockage des photos (candidates + références)")

    cands = list_images(PHOTOS_DIR)
    refs = list_images(REF_DIR)

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Candidates")
        st.write(f"{len(cands)} fichier(s)")
        for fn in cands[:80]:
            with st.expander(fn, expanded=False):
                img = Image.open(os.path.join(PHOTOS_DIR, fn))
                st.image(make_thumb(img, 520), use_container_width=True)
                if st.button("Supprimer", key=f"del_lib_c_{fn}"):
                    try:
                        os.remove(os.path.join(PHOTOS_DIR, fn))
                    except Exception:
                        pass
                    meta = load_meta()
                    meta["photos"].pop(fn, None)
                    save_meta(meta)
                    st.rerun()

    with col2:
        st.write("### Références")
        st.write(f"{len(refs)} fichier(s)")
        for fn in refs[:80]:
            with st.expander(fn, expanded=False):
                img = Image.open(os.path.join(REF_DIR, fn))
                st.image(make_thumb(img, 520), use_container_width=True)
                if st.button("Supprimer", key=f"del_lib_r_{fn}"):
                    try:
                        os.remove(os.path.join(REF_DIR, fn))
                    except Exception:
                        pass
                    meta = load_meta()
                    meta["refs"].pop(fn, None)
                    save_meta(meta)
                    st.rerun()

    st.divider()
    st.caption("⚠️ Note : sur Streamlit Cloud, le stockage local peut être reset lors de certains redeploy. Si tu veux un stockage 100% persistant, il faut une base (Supabase/S3).")


# -------------------------
# TAB 3: FOLLOWERS/FOLLOWING (EXCEL)
# -------------------------
with tab3:
    st.subheader("Abonnés — Comparer Followers vs Following (Excel)")
    st.markdown("""
✅ Upload **1 fichier .xlsx** contenant **2 feuilles** :
- `Followers`
- `Following`

Dans chaque feuille : **colonne A** = usernames (avec ou sans @).  
(Le nom des feuilles peut être `followers`, `FOLLOWERS`, etc. → c’est OK.)
""")

    xls = st.file_uploader("Upload ton fichier Excel (.xlsx)", type=["xlsx"], key="xls_follow")
    if xls:
        try:
            sheets = pd.read_excel(xls, sheet_name=None)
            s_followers = find_sheet(sheets, "Followers")
            s_following = find_sheet(sheets, "Following")

            if not s_followers or not s_following:
                st.error("Je ne trouve pas les feuilles Followers / Following. Vérifie l’orthographe.")
            else:
                followers = extract_usernames_from_sheet(sheets[s_followers])
                following = extract_usernames_from_sheet(sheets[s_following])

                set_fol = set(followers)
                set_ing = set(following)

                not_following_back = sorted(list(set_ing - set_fol))   # tu les suis, ils te suivent pas
                you_dont_follow_back = sorted(list(set_fol - set_ing)) # ils te suivent, tu les suis pas
                mutuals = sorted(list(set_fol & set_ing))

                colA, colB, colC = st.columns(3)
                colA.metric("Followers", len(followers))
                colB.metric("Following", len(following))
                colC.metric("Mutuels", len(mutuals))

                st.write("### 1) Tu suis MAIS ils ne te suivent pas (non-mutuel)")
                st.dataframe(pd.DataFrame({"username": not_following_back}), use_container_width=True)

                st.write("### 2) Ils te suivent MAIS tu ne les suis pas")
                st.dataframe(pd.DataFrame({"username": you_dont_follow_back}), use_container_width=True)

                st.write("### Export")
                out = pd.DataFrame({
                    "not_following_back": pd.Series(not_following_back),
                    "you_dont_follow_back": pd.Series(you_dont_follow_back),
                    "mutuals": pd.Series(mutuals)
                })
                st.download_button(
                    "Télécharger résultat CSV",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="instagram_follow_compare.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error("Erreur lecture Excel. Essaye de ré-enregistrer ton fichier en .xlsx (pas .xls).")


# -------------------------
# TAB 4: POSTS & LIKES (MANUEL)
# -------------------------
with tab4:
    st.subheader("Posts & Likes (manuel) — tu colles les @likers, l’app classe tes fans")

    st.markdown("""
**Étape A :** Upload un fichier `posts.xlsx` ou `posts.csv` avec une colonne obligatoire :
- `post_url`
Optionnel : `date`, `likes_count`

**Étape B :** Pour chaque post, tu colles les usernames qui ont liké (virgules / lignes).
Puis on fait le **Top fans**.
""")

    def read_posts_file(uploaded_file):
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        df.columns = [c.strip().lower() for c in df.columns]
        if "post_url" not in df.columns:
            st.error("Ton fichier doit contenir une colonne 'post_url'.")
            st.stop()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "likes_count" in df.columns:
            df["likes_count"] = pd.to_numeric(df["likes_count"], errors="coerce")
        df = df.dropna(subset=["post_url"]).copy()
        df["post_url"] = df["post_url"].astype(str).str.strip()
        return df

    posts_file = st.file_uploader("Upload posts.xlsx / posts.csv", type=["xlsx", "csv"], key="posts_file")

    interactions = load_interactions()

    if posts_file:
        posts_df = read_posts_file(posts_file)
        st.success(f"{len(posts_df)} post(s) chargés.")

        for idx, row in posts_df.iterrows():
            post_url = row["post_url"]

            col1, col2 = st.columns([2, 3])
            with col1:
                st.markdown(f"**Post #{idx+1}**")
                st.markdown(f"[Ouvrir le post]({post_url})")
                if "date" in posts_df.columns and pd.notnull(row.get("date")):
                    st.caption(f"Date: {row['date']}")
                if "likes_count" in posts_df.columns and pd.notnull(row.get("likes_count")):
                    st.caption(f"Likes (total): {int(row['likes_count'])}")

            with col2:
                existing = interactions[(interactions["post_url"] == post_url) & (interactions["event"] == "like")]
                existing_users = ", ".join(sorted(existing["username"].unique().tolist()))

                text = st.text_area(
                    "Colle les @ qui ont liké ce post (virgules / lignes / espaces)",
                    value=existing_users,
                    key=f"likes_post_{idx}",
                    height=80,
                )

                csave, cclear = st.columns([1, 1])
                with csave:
                    if st.button("Enregistrer", key=f"save_post_{idx}"):
                        users = parse_usernames(text)

                        interactions = interactions[~((interactions["post_url"] == post_url) & (interactions["event"] == "like"))].copy()
                        now = datetime.utcnow().isoformat()
                        to_add = pd.DataFrame({
                            "post_url": [post_url]*len(users),
                            "username": users,
                            "event": ["like"]*len(users),
                            "created_at": [now]*len(users),
                        })
                        interactions = pd.concat([interactions, to_add], ignore_index=True)
                        save_interactions(interactions)
                        st.success(f"{len(users)} like(s) enregistrés.")

                with cclear:
                    if st.button("Vider ce post", key=f"clear_post_{idx}"):
                        interactions = interactions[~((interactions["post_url"] == post_url) & (interactions["event"] == "like"))].copy()
                        save_interactions(interactions)
                        st.warning("Likes supprimés pour ce post.")

            st.divider()

        st.subheader("🏆 Top fans (basé sur tes likes saisis)")
        interactions = load_interactions()
        likes = interactions[interactions["event"] == "like"].copy()

        if likes.empty:
            st.info("Pas encore de likes enregistrés.")
        else:
            top = likes.groupby("username").size().reset_index(name="likes")
            top = top.sort_values("likes", ascending=False)
            st.dataframe(top.head(100), use_container_width=True)

            st.download_button(
                "Télécharger top_fans.csv",
                data=top.to_csv(index=False).encode("utf-8"),
                file_name="top_fans.csv",
                mime="text/csv"
            )

    else:
        st.info("Upload ton fichier posts (urls) pour commencer.")


# -------------------------
# TAB 5: PERFORMANCE (MANUEL / CSV)
# -------------------------
with tab5:
    st.subheader("Performance Instagram (manuel/CSV) — engagement + recommandations + jours actifs")

    st.markdown("""
Tu as 2 options :
- **Manuel** : tu rentres likes/commentaires/reach moyen etc.
- **CSV** : tu upload un fichier avec colonnes : `date`, `likes`, `comments`, `reach` (et optionnel `story_views`)
""")

    subtab_manual, subtab_csv = st.tabs(["✍️ Manuel", "⬆️ CSV"])

    with subtab_manual:
        likes_m = st.number_input("Likes moyens", min_value=0, value=120, step=1)
        com_m = st.number_input("Commentaires moyens", min_value=0, value=6, step=1)
        reach_m = st.number_input("Reach moyen", min_value=0, value=1800, step=50)

        # engagement simple
        engagement = 0.0
        if reach_m > 0:
            engagement = (likes_m + 4 * com_m) / reach_m  # commentaires comptent plus

        st.metric("Taux d'engagement estimé", f"{engagement*100:.2f}%")

        if engagement >= 0.08:
            st.success("Engagement très bon.")
        elif engagement >= 0.05:
            st.info("Engagement correct.")
        else:
            st.warning("Engagement faible → il faut optimiser posts + stories + constance.")

        st.write("### Conseils (alignés avec ton objectif)")
        st.write("• 2 posts / semaine minimum (cohérence > quantité).")
        st.write("• Stories courtes et régulières (1–3/jour).")
        st.write("• Toujours lumière + chaleur + cohérence (beige/sable/sunset).")

    with subtab_csv:
        perf_file = st.file_uploader("Upload performance.csv (date, likes, comments, reach)", type=["csv"], key="perf_csv")
        if perf_file:
            try:
                dfp = pd.read_csv(perf_file)
                dfp.columns = [c.strip().lower() for c in dfp.columns]
                needed = {"date", "likes", "comments", "reach"}
                if not needed.issubset(set(dfp.columns)):
                    st.error("Il manque des colonnes. Il faut au minimum: date, likes, comments, reach")
                else:
                    dfp["date"] = pd.to_datetime(dfp["date"], errors="coerce")
                    dfp = dfp.dropna(subset=["date"]).copy()
                    dfp["likes"] = pd.to_numeric(dfp["likes"], errors="coerce").fillna(0)
                    dfp["comments"] = pd.to_numeric(dfp["comments"], errors="coerce").fillna(0)
                    dfp["reach"] = pd.to_numeric(dfp["reach"], errors="coerce").fillna(0)

                    dfp["engagement"] = np.where(
                        dfp["reach"] > 0,
                        (dfp["likes"] + 4*dfp["comments"]) / dfp["reach"],
                        0
                    )

                    st.write("### Résumé")
                    st.metric("Engagement moyen", f"{dfp['engagement'].mean()*100:.2f}%")
                    st.metric("Reach moyen", f"{dfp['reach'].mean():.0f}")

                    dfp["weekday"] = dfp["date"].dt.day_name()
                    best_day = (
                        dfp.groupby("weekday")["engagement"].mean()
                        .sort_values(ascending=False)
                        .head(1)
                    )
                    st.write("### Jour le plus fort (engagement)")
                    st.dataframe(best_day.reset_index().rename(columns={"engagement":"engagement_moyen"}), use_container_width=True)

                    st.write("### Détails")
                    st.dataframe(dfp.sort_values("date", ascending=False), use_container_width=True)

            except Exception:
                st.error("Erreur de lecture du CSV. Vérifie le format (séparateur virgule).")

st.caption("⚠️ Rappel : IG ne donne pas automatiquement followers/likes via un lien. Ici tout est fait via uploads/saisie (fiable, pas de ban).")
