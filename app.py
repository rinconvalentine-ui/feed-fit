import os
import re
import uuid
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Feed Fit", layout="wide")

DATA_DIR = "data"
PHOTOS_DIR = os.path.join(DATA_DIR, "photos")          # pour feed-fit photos
POSTS_DIR = os.path.join(DATA_DIR, "posts_images")     # pour posts "insta-like"
DB_PATH = os.path.join(DATA_DIR, "feedfit.db")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PHOTOS_DIR, exist_ok=True)
os.makedirs(POSTS_DIR, exist_ok=True)

# =========================
# DB (SQLite) — persistance
# =========================
def db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def db_init():
    conn = db_conn()
    cur = conn.cursor()

    # Posts (grille Insta)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS posts (
        id TEXT PRIMARY KEY,
        image_path TEXT NOT NULL,
        insta_url TEXT,
        post_date TEXT,
        likes_count INTEGER DEFAULT 0,
        created_at TEXT NOT NULL
    );
    """)

    # Likes par post (sélection parmi tes abonnés)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS post_likes (
        post_id TEXT NOT NULL,
        username TEXT NOT NULL,
        PRIMARY KEY (post_id, username),
        FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE
    );
    """)

    # Followers/following sauvegardés (depuis CSV)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ig_users (
        username TEXT PRIMARY KEY,
        is_follower INTEGER DEFAULT 0,
        is_following INTEGER DEFAULT 0,
        updated_at TEXT NOT NULL
    );
    """)

    conn.commit()
    conn.close()

db_init()

# =========================
# HELPERS
# =========================
USERNAME_RE = re.compile(r"^[A-Za-z0-9._]{1,30}$")

def normalize_username(u: str) -> str:
    if u is None:
        return ""
    u = str(u).strip()
    u = u.replace("@", "")
    u = re.sub(r"\s+", "", u)
    u = u.lower()
    return u

def safe_filename(name: str) -> str:
    name = (name or "").strip().replace(" ", "_")
    name = re.sub(r"[^a-zA-Z0-9._-]", "", name)
    return name or uuid.uuid4().hex

def list_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".webp")
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])

def save_uploaded_image(uploaded_file, dest_dir) -> str:
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        ext = ".png"
    fn = f"{safe_filename(os.path.splitext(uploaded_file.name)[0])}_{uuid.uuid4().hex[:10]}{ext}"
    path = os.path.join(dest_dir, fn)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def make_thumb(img: Image.Image, max_side=520):
    img = img.convert("RGB")
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    return img

# =========================
# FEED-FIT SCORING (simple, cohérent)
# =========================
def rgb_to_hsv_np(arr):
    # arr: HxWx3 [0,1]
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

def compute_features(img: Image.Image, max_side=700):
    img = img.convert("RGB")
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))

    arr = np.asarray(img).astype(np.float32) / 255.0
    h, s, v = rgb_to_hsv_np(arr)

    brightness = float(np.mean(v))       # 0..1
    saturation = float(np.mean(s))       # 0..1
    contrast = float(np.std(v))          # 0..0.35+
    warmth = float(np.mean(arr[..., 0] - arr[..., 2]))  # R - B

    # “beige/sable” approx (hue chaud + clair + sat modérée)
    mask_beige = (h > 0.05) & (h < 0.18) & (s > 0.10) & (s < 0.45) & (v > 0.55)
    beige_ratio = float(np.mean(mask_beige))

    return dict(
        brightness=brightness,
        saturation=saturation,
        contrast=contrast,
        warmth=warmth,
        beige_ratio=beige_ratio
    )

def closeness(x, mu, tol):
    d = abs(x - mu)
    return max(0.0, 1.0 - d / max(tol, 1e-6))

def score_photo(feat, target):
    b = closeness(feat["brightness"], target["b_mu"], target["b_tol"])
    s = closeness(feat["saturation"], target["s_mu"], target["s_tol"])
    c = closeness(feat["contrast"],   target["c_mu"], target["c_tol"])
    w = closeness(feat["warmth"],     target["w_mu"], target["w_tol"])
    z = closeness(feat["beige_ratio"],target["z_mu"], target["z_tol"])

    score = (
        target["wb"]*b + target["ws"]*s + target["wc"]*c + target["ww"]*w + target["wz"]*z
    ) / (target["wb"]+target["ws"]+target["wc"]+target["ww"]+target["wz"]+1e-9)

    return int(round(score * 100))

def advice(feat, target):
    out = []
    if feat["brightness"] < target["b_mu"] - target["b_tol"]*0.35:
        out.append("Trop sombre → monte exposition / hautes lumières.")
    if feat["brightness"] > target["b_mu"] + target["b_tol"]*0.35:
        out.append("Trop clair → baisse hautes lumières / blancs.")
    if feat["saturation"] > target["s_mu"] + target["s_tol"]*0.35:
        out.append("Trop saturé → baisse saturation (-5 à -15).")
    if feat["saturation"] < target["s_mu"] - target["s_tol"]*0.35:
        out.append("Trop terne → + légère saturation (+3 à +8).")
    if feat["contrast"] > target["c_mu"] + target["c_tol"]*0.35:
        out.append("Contraste dur → adoucir contraste / noirs.")
    if feat["warmth"] < target["w_mu"] - target["w_tol"]*0.35:
        out.append("Trop froid → réchauffer la température.")
    if feat["beige_ratio"] < target["z_mu"] - target["z_tol"]*0.35:
        out.append("Pas assez sable/beige → privilégie tons chauds clairs (sable, crème, sunset).")
    if not out:
        out.append("Cohérent avec ta cible → micro-ajustements seulement.")
    return out

# =========================
# FOLLOWERS/FOLLOWING (CSV) -> DB
# =========================
def read_single_column_csv(file) -> list[str]:
    df = pd.read_csv(file, header=None)
    s = df.iloc[:, 0].astype(str).str.strip()
    s = s[s != ""]
    s = s.str.replace("@", "", regex=False).str.lower()
    s = s[s != "nan"]
    usernames = []
    for x in s.tolist():
        x = normalize_username(x)
        if x and USERNAME_RE.match(x):
            usernames.append(x)
    # dedupe preserve order
    seen = set()
    out = []
    for u in usernames:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def save_follow_state(followers: list[str], following: list[str]):
    conn = db_conn()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()

    # reset flags
    cur.execute("UPDATE ig_users SET is_follower = 0, is_following = 0, updated_at = ?", (now,))

    # upsert followers
    for u in followers:
        cur.execute("""
        INSERT INTO ig_users(username, is_follower, is_following, updated_at)
        VALUES(?, 1, 0, ?)
        ON CONFLICT(username) DO UPDATE SET is_follower=1, updated_at=excluded.updated_at
        """, (u, now))

    # upsert following
    for u in following:
        cur.execute("""
        INSERT INTO ig_users(username, is_follower, is_following, updated_at)
        VALUES(?, 0, 1, ?)
        ON CONFLICT(username) DO UPDATE SET is_following=1, updated_at=excluded.updated_at
        """, (u, now))

    conn.commit()
    conn.close()

def get_follow_lists_from_db():
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT username, is_follower, is_following FROM ig_users")
    rows = cur.fetchall()
    conn.close()

    followers = [r[0] for r in rows if int(r[1]) == 1]
    following = [r[0] for r in rows if int(r[2]) == 1]
    return sorted(followers), sorted(following)

# =========================
# POSTS (Insta grid) + likers persistence
# =========================
def add_post(image_path, insta_url, post_date, likes_count):
    conn = db_conn()
    cur = conn.cursor()
    pid = uuid.uuid4().hex
    cur.execute(
        "INSERT INTO posts(id, image_path, insta_url, post_date, likes_count, created_at) VALUES(?,?,?,?,?,?)",
        (pid, image_path, insta_url or "", post_date or "", int(likes_count or 0), datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()
    return pid

def get_posts():
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, image_path, insta_url, post_date, likes_count FROM posts ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    return [
        dict(id=r[0], image_path=r[1], insta_url=r[2] or "", post_date=r[3] or "", likes_count=int(r[4] or 0))
        for r in rows
    ]

def delete_post(post_id):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT image_path FROM posts WHERE id = ?", (post_id,))
    row = cur.fetchone()
    cur.execute("DELETE FROM posts WHERE id = ?", (post_id,))
    conn.commit()
    conn.close()
    if row and row[0] and os.path.exists(row[0]):
        try:
            os.remove(row[0])
        except Exception:
            pass

def update_post_meta(post_id, insta_url, post_date, likes_count):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("UPDATE posts SET insta_url=?, post_date=?, likes_count=? WHERE id=?",
                (insta_url or "", post_date or "", int(likes_count or 0), post_id))
    conn.commit()
    conn.close()

def get_likers(post_id) -> list[str]:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT username FROM post_likes WHERE post_id=? ORDER BY username ASC", (post_id,))
    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]

def set_likers(post_id, usernames: list[str]):
    usernames = [normalize_username(u) for u in usernames if normalize_username(u)]
    usernames = sorted(set([u for u in usernames if USERNAME_RE.match(u)]))

    conn = db_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM post_likes WHERE post_id=?", (post_id,))
    for u in usernames:
        cur.execute("INSERT OR IGNORE INTO post_likes(post_id, username) VALUES(?,?)", (post_id, u))
    conn.commit()
    conn.close()

def top_fans():
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT username, COUNT(*) as likes
        FROM post_likes
        GROUP BY username
        ORDER BY likes DESC
        LIMIT 200
    """)
    rows = cur.fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=["username", "likes"])

# =========================
# UI
# =========================
st.title("Feed Fit — Photos + Posts (3×3) + Followers/Following + Performance")

tab_feed, tab_posts, tab_follow, tab_perf, tab_storage = st.tabs([
    "📸 Feed Fit (photos)",
    "🧾 Posts (style Insta 3×3)",
    "👥 Followers/Following (CSV)",
    "📈 Performance",
    "🗂️ Stockage"
])

# --------------------------------
# TAB: FOLLOWERS/FOLLOWING (CSV)
# --------------------------------
with tab_follow:
    st.subheader("Followers/Following — Upload 2 CSV (fiable)")

    st.markdown("""
Tu uploades :
- **followers.csv** (à gauche)
- **following.csv** (à droite)

Ensuite l’app calcule :
- qui ne te suit pas en retour
- mutuals
- qui te suit mais que tu ne suis pas
Et **sauvegarde** la liste pour les autres onglets (notamment Posts).
""")

    col1, col2 = st.columns(2)
    with col1:
        followers_csv = st.file_uploader("Upload followers.csv", type=["csv"], key="followers_csv")
    with col2:
        following_csv = st.file_uploader("Upload following.csv", type=["csv"], key="following_csv")

    if followers_csv and following_csv:
        followers = read_single_column_csv(followers_csv)
        following = read_single_column_csv(following_csv)

        save_follow_state(followers, following)
        st.success("Listes enregistrées ✅ (stockées pour toute l’app)")

        set_fol = set(followers)
        set_ing = set(following)

        not_following_back = sorted(list(set_ing - set_fol))   # tu suis, ils te suivent pas
        you_dont_follow_back = sorted(list(set_fol - set_ing)) # ils te suivent, tu les suis pas
        mutuals = sorted(list(set_fol & set_ing))

        a, b, c = st.columns(3)
        a.metric("Followers", len(followers))
        b.metric("Following", len(following))
        c.metric("Mutuels", len(mutuals))

        st.write("### ❌ Tu suis MAIS ils ne te suivent pas")
        st.dataframe(pd.DataFrame({"username": not_following_back}), use_container_width=True)

        st.write("### ✅ Mutuals")
        st.dataframe(pd.DataFrame({"username": mutuals}), use_container_width=True)

        st.write("### ➕ Ils te suivent MAIS tu ne les suis pas")
        st.dataframe(pd.DataFrame({"username": you_dont_follow_back}), use_container_width=True)

        out = pd.DataFrame({
            "not_following_back": pd.Series(not_following_back),
            "mutuals": pd.Series(mutuals),
            "you_dont_follow_back": pd.Series(you_dont_follow_back),
        })
        st.download_button(
            "⬇️ Télécharger le résultat",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="follow_analysis.csv",
            mime="text/csv"
        )
    else:
        st.info("Upload les 2 CSV pour lancer l’analyse.")

# On récupère la liste followers depuis DB (pour Posts)
followers_db, following_db = get_follow_lists_from_db()

# --------------------------------
# TAB: POSTS (3×3 like Insta)
# --------------------------------
with tab_posts:
    st.subheader("Posts — affichage grille 3×3 + likes sauvegardés")

    st.markdown("""
✅ Tu ajoutes tes posts **une fois** (photo + lien + date + likes_count).  
✅ Tu cliques sur un post → tu coches **dans tes followers** qui a liké.  
✅ Tout est **sauvegardé** (tu ne recommences jamais).
""")

    with st.expander("➕ Ajouter un post", expanded=False):
        up = st.file_uploader("Upload photo du post", type=["jpg", "jpeg", "png", "webp"], key="add_post_img")
        colA, colB = st.columns(2)
        with colA:
            insta_url = st.text_input("Lien Instagram (optionnel)", placeholder="https://www.instagram.com/p/....", key="add_post_url")
            post_date = st.text_input("Date (optionnel)", placeholder="2025-12-20", key="add_post_date")
        with colB:
            likes_count = st.number_input("Nombre de likes (optionnel)", min_value=0, step=1, value=0, key="add_post_likes")

        if st.button("Ajouter au feed", type="primary", disabled=(up is None)):
            img_path = save_uploaded_image(up, POSTS_DIR)
            add_post(img_path, insta_url, post_date, likes_count)
            st.success("Post ajouté ✅")
            st.rerun()

    posts = get_posts()
    if "selected_post_id" not in st.session_state:
        st.session_state.selected_post_id = None

    if not posts:
        st.info("Ajoute tes posts pour remplir la grille.")
    else:
        st.markdown("### 📸 Feed (clique sur un post)")
        cols = st.columns(3)
        for i, p in enumerate(posts):
            with cols[i % 3]:
                try:
                    st.image(p["image_path"], use_container_width=True)
                except Exception:
                    st.warning("Image illisible (supprime et ré-uploade).")
                if st.button("Ouvrir", key=f"open_{p['id']}"):
                    st.session_state.selected_post_id = p["id"]
                    st.rerun()

        pid = st.session_state.selected_post_id
        if pid:
            post_map = {x["id"]: x for x in posts}
            if pid not in post_map:
                st.session_state.selected_post_id = None
                st.rerun()

            p = post_map[pid]
            st.divider()
            st.markdown("### 🧾 Détails du post sélectionné")

            left, right = st.columns([1, 1])
            with left:
                st.image(p["image_path"], use_container_width=True)
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("🗑️ Supprimer", key=f"del_{pid}"):
                        delete_post(pid)
                        st.session_state.selected_post_id = None
                        st.success("Supprimé ✅")
                        st.rerun()
                with c2:
                    if st.button("⬅️ Fermer", key=f"close_{pid}"):
                        st.session_state.selected_post_id = None
                        st.rerun()

            with right:
                insta_url2 = st.text_input("Lien Instagram", value=p["insta_url"], key=f"url_{pid}")
                post_date2 = st.text_input("Date", value=p["post_date"], key=f"date_{pid}")
                likes_count2 = st.number_input("Nombre de likes", min_value=0, step=1, value=int(p["likes_count"]), key=f"likes_{pid}")

                if st.button("Enregistrer infos", type="primary", key=f"save_meta_{pid}"):
                    update_post_meta(pid, insta_url2, post_date2, likes_count2)
                    st.success("Infos enregistrées ✅")
                    st.rerun()

                st.markdown("#### ❤️ Sélection des abonnés qui ont liké")
                if not followers_db:
                    st.warning("Tu n’as pas encore chargé tes followers. Va dans l’onglet Followers/Following et upload tes 2 CSV.")
                else:
                    current = get_likers(pid)
                    picked = st.multiselect(
                        "Coche ceux qui ont liké (recherche possible)",
                        options=followers_db,
                        default=[u for u in current if u in followers_db],
                        key=f"likers_{pid}"
                    )
                    if st.button("Enregistrer likes", key=f"save_likes_{pid}"):
                        set_likers(pid, picked)
                        st.success("Likes enregistrés ✅")
                        st.rerun()

                    st.caption("Astuce : tape 3 lettres dans la recherche pour aller vite.")

            # Résumé
            likers = get_likers(pid)
            st.markdown("#### 📌 Résumé")
            st.write(f"- Likers cochés : **{len(likers)}**")
            st.write(f"- Likes total indiqué : **{likes_count2}**")

        st.divider()
        st.subheader("🏆 Top fans (sur l’ensemble des posts)")
        df_top = top_fans()
        if df_top.empty:
            st.info("Pas encore de likes cochés.")
        else:
            st.dataframe(df_top.head(50), use_container_width=True)
            st.download_button(
                "⬇️ Télécharger top_fans.csv",
                data=df_top.to_csv(index=False).encode("utf-8"),
                file_name="top_fans.csv",
                mime="text/csv"
            )

# --------------------------------
# TAB: FEED FIT (PHOTOS) — stockage + conseils
# --------------------------------
with tab_feed:
    st.subheader("Feed Fit — sélection photos (top9 + conseils)")

    st.markdown("""
Tu uploades des photos candidates → elles sont **stockées**.  
L’app te sort un **Top 9** + conseils pour coller à ton feed :
**lumineux / chaud / sable-beige / luxe discret**.
""")

    with st.expander("🎯 Réglages de cible (feed)", expanded=True):
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
            st.caption("Pondérations")
            wb = st.slider("Poids luminosité", 1, 10, 7)
            ws = st.slider("Poids saturation", 1, 10, 5)
            wc = st.slider("Poids contraste", 1, 10, 4)
            ww = st.slider("Poids chaleur", 1, 10, 6)
            wz = st.slider("Poids sable/beige", 1, 10, 6)

    target = dict(
        b_mu=b_mu, b_tol=b_tol,
        s_mu=s_mu, s_tol=s_tol,
        c_mu=c_mu, c_tol=c_tol,
        w_mu=w_mu, w_tol=w_tol,
        z_mu=z_mu, z_tol=z_tol,
        wb=wb, ws=ws, wc=wc, ww=ww, wz=wz
    )

    up = st.file_uploader("📷 Upload photos candidates", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
    if up:
        for f in up:
            pth = save_uploaded_image(f, PHOTOS_DIR)
        st.success("Photos ajoutées au stockage ✅")
        st.rerun()

    files = list_images(PHOTOS_DIR)
    if not files:
        st.info("Ajoute des photos candidates pour commencer.")
    else:
        results = []
        for fn in files:
            path = os.path.join(PHOTOS_DIR, fn)
            try:
                img = Image.open(path)
                feat = compute_features(img)
                sc = score_photo(feat, target)
                tips = advice(feat, target)
                results.append({
                    "filename": fn,
                    "score": sc,
                    "feat": feat,
                    "tips": tips
                })
            except Exception:
                pass

        df = pd.DataFrame(results).sort_values("score", ascending=False)
        st.subheader("Top 9")
        top9 = df.head(9)

        cols = st.columns(3)
        for i, row in enumerate(top9.itertuples(index=False)):
            with cols[i % 3]:
                img = Image.open(os.path.join(PHOTOS_DIR, row.filename))
                st.image(make_thumb(img, 520), caption=f"{row.filename} — {row.score}/100", use_container_width=True)

        st.divider()
        st.subheader("Détails + conseils (par photo)")

        for row in df.itertuples(index=False):
            with st.expander(f"{row.score}/100 — {row.filename}", expanded=False):
                img = Image.open(os.path.join(PHOTOS_DIR, row.filename))
                st.image(make_thumb(img, 760), use_container_width=True)
                for t in row.tips:
                    st.write("• " + t)
                f = row.feat
                st.caption(
                    f"lum={f['brightness']:.3f} | sat={f['saturation']:.3f} | contrast={f['contrast']:.3f} | warm={f['warmth']:.3f} | beige={f['beige_ratio']:.3f}"
                )
                if st.button("🗑️ Supprimer cette photo", key=f"del_feed_{row.filename}"):
                    try:
                        os.remove(os.path.join(PHOTOS_DIR, row.filename))
                    except Exception:
                        pass
                    st.rerun()

# --------------------------------
# TAB: PERFORMANCE
# --------------------------------
with tab_perf:
    st.subheader("Performance — manuel ou CSV")

    sub_manual, sub_csv = st.tabs(["✍️ Manuel", "⬆️ CSV"])

    with sub_manual:
        likes_m = st.number_input("Likes moyens", min_value=0, value=120, step=1)
        com_m = st.number_input("Commentaires moyens", min_value=0, value=6, step=1)
        reach_m = st.number_input("Reach moyen", min_value=0, value=1800, step=50)

        eng = 0.0
        if reach_m > 0:
            eng = (likes_m + 4 * com_m) / reach_m

        st.metric("Engagement estimé", f"{eng*100:.2f}%")

        if eng >= 0.08:
            st.success("Très bon.")
        elif eng >= 0.05:
            st.info("Correct.")
        else:
            st.warning("Faible → optimiser constance + contenu + stories.")

        st.write("Conseils rapides :")
        st.write("• 2 posts/semaine (cohérence > quantité)")
        st.write("• stories régulières")
        st.write("• lumière + chaleur + sable/beige")

    with sub_csv:
        st.markdown("CSV attendu : colonnes `date, likes, comments, reach` (optionnel : `saves, shares`)")
        perf = st.file_uploader("Upload performance.csv", type=["csv"])
        if perf:
            dfp = pd.read_csv(perf)
            dfp.columns = [c.strip().lower() for c in dfp.columns]
            needed = {"date", "likes", "comments", "reach"}
            if not needed.issubset(set(dfp.columns)):
                st.error("Il manque des colonnes. Il faut au minimum : date, likes, comments, reach")
            else:
                dfp["date"] = pd.to_datetime(dfp["date"], errors="coerce")
                dfp = dfp.dropna(subset=["date"]).copy()
                for c in ["likes", "comments", "reach", "saves", "shares"]:
                    if c in dfp.columns:
                        dfp[c] = pd.to_numeric(dfp[c], errors="coerce").fillna(0)

                saves = dfp["saves"] if "saves" in dfp.columns else 0
                shares = dfp["shares"] if "shares" in dfp.columns else 0

                dfp["engagement"] = np.where(
                    dfp["reach"] > 0,
                    (dfp["likes"] + 4*dfp["comments"] + 2*saves + 2*shares) / dfp["reach"],
                    0
                )

                st.metric("Engagement moyen", f"{dfp['engagement'].mean()*100:.2f}%")
                dfp["weekday"] = dfp["date"].dt.day_name()
                best_days = dfp.groupby("weekday")["engagement"].mean().sort_values(ascending=False)
                st.write("### Meilleurs jours (engagement moyen)")
                st.dataframe(best_days, use_container_width=True)
                st.write("### Top posts")
                st.dataframe(dfp.sort_values("engagement", ascending=False).head(15), use_container_width=True)

# --------------------------------
# TAB: STOCKAGE
# --------------------------------
with tab_storage:
    st.subheader("Stockage — voir / supprimer")

    c1, c2 = st.columns(2)
    with c1:
        st.write("### Photos (Feed Fit)")
        feed_files = list_images(PHOTOS_DIR)
        st.write(f"{len(feed_files)} fichier(s)")
        for fn in feed_files[:60]:
            with st.expander(fn, expanded=False):
                st.image(make_thumb(Image.open(os.path.join(PHOTOS_DIR, fn)), 520), use_container_width=True)
                if st.button("Supprimer", key=f"del_store_feed_{fn}"):
                    try:
                        os.remove(os.path.join(PHOTOS_DIR, fn))
                    except Exception:
                        pass
                    st.rerun()

    with c2:
        st.write("### Posts (grille Insta)")
        posts = get_posts()
        st.write(f"{len(posts)} post(s)")
        for p in posts[:60]:
            with st.expander(p["id"], expanded=False):
                st.image(make_thumb(Image.open(p["image_path"]), 520), use_container_width=True)
                st.caption(p.get("insta_url",""))
                if st.button("Supprimer ce post", key=f"del_store_post_{p['id']}"):
                    delete_post(p["id"])
                    st.rerun()

    st.divider()
    st.caption("⚠️ Si tu es sur Streamlit Cloud, le stockage local peut parfois être réinitialisé lors de certains redeploy.")
