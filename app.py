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
st.set_page_config(page_title="Feed Fit — Planner", layout="wide")

DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "feedfit.db")

FOLLOW_DIR = os.path.join(DATA_DIR, "followers")       # pas obligatoire, juste organisation
POSTS_DIR = os.path.join(DATA_DIR, "posts_images")     # posts “style insta”
PLANNER_DIR = os.path.join(DATA_DIR, "planner_images") # images candidates pour planner

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FOLLOW_DIR, exist_ok=True)
os.makedirs(POSTS_DIR, exist_ok=True)
os.makedirs(PLANNER_DIR, exist_ok=True)

USERNAME_RE = re.compile(r"^[A-Za-z0-9._]{1,30}$")

# =========================
# DB
# =========================
def db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def db_init():
    conn = db_conn()
    cur = conn.cursor()

    # Followers/following sauvegardés
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ig_users (
        username TEXT PRIMARY KEY,
        is_follower INTEGER DEFAULT 0,
        is_following INTEGER DEFAULT 0,
        updated_at TEXT NOT NULL
    );
    """)

    # Posts “Insta”
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

    # Likers par post
    cur.execute("""
    CREATE TABLE IF NOT EXISTS post_likes (
        post_id TEXT NOT NULL,
        username TEXT NOT NULL,
        PRIMARY KEY (post_id, username),
        FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE
    );
    """)

    # Planner items (candidates)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS planner_photos (
        id TEXT PRIMARY KEY,
        image_path TEXT NOT NULL,
        original_name TEXT,
        width INTEGER,
        height INTEGER,
        created_at TEXT NOT NULL
    );
    """)

    conn.commit()
    conn.close()

db_init()

# =========================
# HELPERS
# =========================
def normalize_username(u: str) -> str:
    if u is None: 
        return ""
    u = str(u).strip().replace("@", "")
    u = re.sub(r"\s+", "", u)
    return u.lower()

def safe_filename(name: str) -> str:
    name = (name or "").strip().replace(" ", "_")
    name = re.sub(r"[^a-zA-Z0-9._-]", "", name)
    return name or uuid.uuid4().hex

def make_thumb(img: Image.Image, max_side=520):
    img = img.convert("RGB")
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    return img

def save_uploaded_image(uploaded_file, dest_dir) -> str:
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        ext = ".png"
    fn = f"{safe_filename(os.path.splitext(uploaded_file.name)[0])}_{uuid.uuid4().hex[:10]}{ext}"
    path = os.path.join(dest_dir, fn)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def read_single_column_csv(file) -> list[str]:
    df = pd.read_csv(file, header=None)
    s = df.iloc[:, 0].astype(str).str.strip()
    s = s[s != ""].str.replace("@", "", regex=False).str.lower()
    s = s[s != "nan"]
    out = []
    seen = set()
    for x in s.tolist():
        x = normalize_username(x)
        if x and USERNAME_RE.match(x) and x not in seen:
            seen.add(x)
            out.append(x)
    return out

# =========================
# FOLLOWERS/FOLLOWING -> DB
# =========================
def save_follow_state(followers: list[str], following: list[str]):
    conn = db_conn()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()

    # reset flags
    cur.execute("UPDATE ig_users SET is_follower=0, is_following=0, updated_at=?", (now,))

    for u in followers:
        cur.execute("""
        INSERT INTO ig_users(username, is_follower, is_following, updated_at)
        VALUES(?, 1, 0, ?)
        ON CONFLICT(username) DO UPDATE SET is_follower=1, updated_at=excluded.updated_at
        """, (u, now))

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
    followers = sorted([r[0] for r in rows if int(r[1]) == 1])
    following = sorted([r[0] for r in rows if int(r[2]) == 1])
    return followers, following

# =========================
# POSTS (Insta) + likers
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
    return [dict(id=r[0], image_path=r[1], insta_url=r[2] or "", post_date=r[3] or "", likes_count=int(r[4] or 0)) for r in rows]

def delete_post(post_id):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT image_path FROM posts WHERE id=?", (post_id,))
    row = cur.fetchone()
    cur.execute("DELETE FROM posts WHERE id=?", (post_id,))
    conn.commit()
    conn.close()
    if row and row[0] and os.path.exists(row[0]):
        try: os.remove(row[0])
        except: pass

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
    usernames = sorted(set([normalize_username(u) for u in usernames if USERNAME_RE.match(normalize_username(u))]))
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
# FEED PLANNER (auto 3×3)
# =========================
def rgb_to_hsv_np(arr):
    r, g, b = arr[...,0], arr[...,1], arr[...,2]
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

def compute_features(img: Image.Image, max_side=900):
    img = img.convert("RGB")
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w*scale), int(h*scale)))
    arr = np.asarray(img).astype(np.float32) / 255.0
    hch, sch, vch = rgb_to_hsv_np(arr)

    brightness = float(np.mean(vch))
    saturation = float(np.mean(sch))
    contrast = float(np.std(vch))
    warmth = float(np.mean(arr[...,0] - arr[...,2]))

    mask_beige = (hch > 0.05) & (hch < 0.18) & (sch > 0.10) & (sch < 0.45) & (vch > 0.55)
    beige_ratio = float(np.mean(mask_beige))

    return dict(brightness=brightness, saturation=saturation, contrast=contrast, warmth=warmth, beige_ratio=beige_ratio)

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

    return int(round(score*100))

def advice(feat, target):
    out=[]
    if feat["brightness"] < target["b_mu"] - target["b_tol"]*0.35: out.append("Trop sombre → monte exposition / HL.")
    if feat["brightness"] > target["b_mu"] + target["b_tol"]*0.35: out.append("Trop clair → baisse HL / blancs.")
    if feat["saturation"] > target["s_mu"] + target["s_tol"]*0.35: out.append("Trop saturé → baisse saturation.")
    if feat["saturation"] < target["s_mu"] - target["s_tol"]*0.35: out.append("Trop terne → + légère saturation.")
    if feat["warmth"] < target["w_mu"] - target["w_tol"]*0.35: out.append("Trop froid → réchauffer la température.")
    if feat["beige_ratio"] < target["z_mu"] - target["z_tol"]*0.35: out.append("Pas assez sable/beige → privilégie tons chauds clairs.")
    if not out: out.append("Cohérent → micro-ajustements seulement.")
    return out

def planner_add_photo(path, original_name, w, h):
    conn = db_conn()
    cur = conn.cursor()
    pid = uuid.uuid4().hex
    cur.execute(
        "INSERT INTO planner_photos(id, image_path, original_name, width, height, created_at) VALUES(?,?,?,?,?,?)",
        (pid, path, original_name or "", int(w), int(h), datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

def planner_get_photos():
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, image_path, original_name, width, height FROM planner_photos ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    return [dict(id=r[0], image_path=r[1], original_name=r[2], width=int(r[3]), height=int(r[4])) for r in rows]

def planner_delete(photo_id):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT image_path FROM planner_photos WHERE id=?", (photo_id,))
    row = cur.fetchone()
    cur.execute("DELETE FROM planner_photos WHERE id=?", (photo_id,))
    conn.commit()
    conn.close()
    if row and row[0] and os.path.exists(row[0]):
        try: os.remove(row[0])
        except: pass

def auto_place_grid(items):
    """
    items: list of dict with score + features + path
    simple placement:
      - sort by score desc
      - alternate brightness/warmth to avoid clumps
    returns list of 9 items in grid order (row-major)
    """
    if len(items) <= 9:
        chosen = sorted(items, key=lambda x: x["score"], reverse=True)
    else:
        chosen = sorted(items, key=lambda x: x["score"], reverse=True)[:9]

    # split bright vs dark
    bright = sorted(chosen, key=lambda x: x["feat"]["brightness"], reverse=True)
    dark = list(reversed(bright))

    # positions: center important, then corners, then edges
    pos_order = [4, 0, 2, 6, 8, 1, 3, 5, 7]

    grid = [None]*9
    # alternate from bright and dark for balance
    take_from_bright = True
    for p in pos_order:
        if take_from_bright and bright:
            grid[p] = bright.pop(0)
        elif dark:
            grid[p] = dark.pop(0)
        else:
            grid[p] = bright.pop(0) if bright else None
        take_from_bright = not take_from_bright

    # fill any None
    rest = [x for x in chosen if x not in grid]
    for i in range(9):
        if grid[i] is None and rest:
            grid[i] = rest.pop(0)
    return grid

# =========================
# UI
# =========================
st.title("Feed Fit — Planner Instagram + Analyse")

tab_planner, tab_posts, tab_follow, tab_perf, tab_storage = st.tabs([
    "🧩 Feed Planner (3×3 auto)",
    "🧾 Posts (style Insta 3×3)",
    "👥 Followers/Following (CSV)",
    "📈 Performance",
    "🗂️ Stockage"
])

# -------------------------
# FOLLOW TAB (CSV)
# -------------------------
with tab_follow:
    st.subheader("Upload 2 CSV : followers + following")
    c1, c2 = st.columns(2)
    with c1:
        followers_csv = st.file_uploader("followers.csv", type=["csv"])
    with c2:
        following_csv = st.file_uploader("following.csv", type=["csv"])

    if followers_csv and following_csv:
        followers = read_single_column_csv(followers_csv)
        following = read_single_column_csv(following_csv)
        save_follow_state(followers, following)
        st.success("Listes sauvegardées ✅")

        set_fol = set(followers)
        set_ing = set(following)

        not_following_back = sorted(list(set_ing - set_fol))
        you_dont_follow_back = sorted(list(set_fol - set_ing))
        mutuals = sorted(list(set_fol & set_ing))

        a,b,c = st.columns(3)
        a.metric("Followers", len(followers))
        b.metric("Following", len(following))
        c.metric("Mutuels", len(mutuals))

        st.write("### ❌ Tu suis mais ils ne te suivent pas")
        st.dataframe(pd.DataFrame({"username": not_following_back}), use_container_width=True)

        st.write("### ✅ Mutuals")
        st.dataframe(pd.DataFrame({"username": mutuals}), use_container_width=True)

        st.write("### ➕ Ils te suivent mais tu ne les suis pas")
        st.dataframe(pd.DataFrame({"username": you_dont_follow_back}), use_container_width=True)

        out = pd.DataFrame({
            "not_following_back": pd.Series(not_following_back),
            "mutuals": pd.Series(mutuals),
            "you_dont_follow_back": pd.Series(you_dont_follow_back),
        })
        st.download_button("Télécharger analyse", out.to_csv(index=False).encode("utf-8"),
                           file_name="follow_analysis.csv", mime="text/csv")
    else:
        st.info("Upload les 2 CSV pour analyser.")

followers_db, following_db = get_follow_lists_from_db()

# -------------------------
# POSTS TAB (Insta 3×3)
# -------------------------
with tab_posts:
    st.subheader("Posts (grille 3×3) + likes par abonnés (sauvegardé)")

    with st.expander("➕ Ajouter un post", expanded=False):
        up = st.file_uploader("Photo du post", type=["jpg","jpeg","png","webp"], key="post_img")
        colA, colB = st.columns(2)
        with colA:
            insta_url = st.text_input("Lien Instagram (optionnel)", key="post_url")
            post_date = st.text_input("Date (optionnel)", placeholder="2025-12-23", key="post_date")
        with colB:
            likes_count = st.number_input("Nombre de likes", min_value=0, value=0, step=1, key="post_likecount")

        if st.button("Ajouter", type="primary", disabled=(up is None)):
            path = save_uploaded_image(up, POSTS_DIR)
            add_post(path, insta_url, post_date, likes_count)
            st.success("Ajouté ✅")
            st.rerun()

    posts = get_posts()
    if "selected_post_id" not in st.session_state:
        st.session_state.selected_post_id = None

    if not posts:
        st.info("Ajoute des posts pour remplir la grille.")
    else:
        st.markdown("### Grille (clique sur un post)")
        cols = st.columns(3)
        for i,p in enumerate(posts):
            with cols[i%3]:
                st.image(p["image_path"], use_container_width=True)
                st.caption(f"❤️ {p['likes_count']}")
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
            left,right = st.columns([1,1])
            with left:
                st.image(p["image_path"], use_container_width=True)
                c1,c2 = st.columns(2)
                with c1:
                    if st.button("🗑️ Supprimer", key=f"del_{pid}"):
                        delete_post(pid)
                        st.session_state.selected_post_id = None
                        st.rerun()
                with c2:
                    if st.button("Fermer", key=f"close_{pid}"):
                        st.session_state.selected_post_id = None
                        st.rerun()

            with right:
                insta_url2 = st.text_input("Lien Instagram", value=p["insta_url"], key=f"url_{pid}")
                post_date2 = st.text_input("Date", value=p["post_date"], key=f"date_{pid}")
                likes_count2 = st.number_input("Likes", min_value=0, value=int(p["likes_count"]), step=1, key=f"likes_{pid}")

                if st.button("Enregistrer infos", type="primary", key=f"save_meta_{pid}"):
                    update_post_meta(pid, insta_url2, post_date2, likes_count2)
                    st.success("Sauvé ✅")
                    st.rerun()

                st.markdown("#### Qui a liké (parmi tes followers)")
                if not followers_db:
                    st.warning("Charge d’abord tes followers (onglet Followers/Following).")
                else:
                    current = get_likers(pid)
                    picked = st.multiselect(
                        "Sélectionne les abonnés qui ont liké",
                        options=followers_db,
                        default=[u for u in current if u in followers_db],
                        key=f"likers_{pid}"
                    )
                    if st.button("Enregistrer likes", key=f"save_likes_{pid}"):
                        set_likers(pid, picked)
                        st.success("Likes enregistrés ✅")
                        st.rerun()

        st.divider()
        st.subheader("🏆 Top fans")
        df_top = top_fans()
        if df_top.empty:
            st.info("Pas encore de likes cochés.")
        else:
            st.dataframe(df_top.head(50), use_container_width=True)
            st.download_button("Télécharger top_fans.csv",
                               df_top.to_csv(index=False).encode("utf-8"),
                               file_name="top_fans.csv", mime="text/csv")

# -------------------------
# PLANNER TAB (auto 3×3)
# -------------------------
with tab_planner:
    st.subheader("Feed Planner — tu uploades, l’app choisit + place en 3×3")

    # Qualité minimale
    st.markdown("### Qualité (filtre)")
    min_w = st.number_input("Largeur minimale (px)", min_value=0, value=3840, step=10)
    min_h = st.number_input("Hauteur minimale (px)", min_value=0, value=2160, step=10)
    st.caption("👉 Si tes photos ne sont pas au moins à cette taille, elles seront refusées (pas de “fausse 4K”).")

    # Target feed
    with st.expander("🎯 Cible du feed (lumière / beige / chaleur)", expanded=True):
        colA, colB, colC = st.columns(3)
        with colA:
            b_mu = st.slider("Luminosité cible", 0.35, 0.85, 0.68, 0.01)
            b_tol = st.slider("Tolérance luminosité", 0.05, 0.30, 0.16, 0.01)
            c_mu = st.slider("Contraste cible", 0.02, 0.20, 0.08, 0.005)
            c_tol = st.slider("Tolérance contraste", 0.01, 0.20, 0.06, 0.005)
        with colB:
            s_mu = st.slider("Saturation cible", 0.05, 0.70, 0.28, 0.01)
            s_tol = st.slider("Tolérance saturation", 0.05, 0.50, 0.18, 0.01)
            w_mu = st.slider("Chaleur cible", -0.15, 0.25, 0.06, 0.01)
            w_tol = st.slider("Tolérance chaleur", 0.05, 0.40, 0.16, 0.01)
        with colC:
            z_mu = st.slider("Beige/sable cible", 0.00, 0.45, 0.10, 0.01)
            z_tol = st.slider("Tolérance beige/sable", 0.02, 0.50, 0.10, 0.01)
            wb = st.slider("Poids luminosité", 1, 10, 7)
            ws = st.slider("Poids saturation", 1, 10, 5)
            wc = st.slider("Poids contraste", 1, 10, 4)
            ww = st.slider("Poids chaleur", 1, 10, 6)
            wz = st.slider("Poids beige/sable", 1, 10, 6)

    target = dict(b_mu=b_mu, b_tol=b_tol, s_mu=s_mu, s_tol=s_tol, c_mu=c_mu, c_tol=c_tol,
                  w_mu=w_mu, w_tol=w_tol, z_mu=z_mu, z_tol=z_tol, wb=wb, ws=ws, wc=wc, ww=ww, wz=wz)

    st.divider()

    up = st.file_uploader("Upload photos candidates (Planner)", type=["jpg","jpeg","png","webp"], accept_multiple_files=True)
    if up:
        accepted = 0
        refused = 0
        for f in up:
            path = save_uploaded_image(f, PLANNER_DIR)
            try:
                img = Image.open(path)
                w,h = img.size
                if w < int(min_w) or h < int(min_h):
                    refused += 1
                    os.remove(path)
                    continue
                planner_add_photo(path, f.name, w, h)
                accepted += 1
            except Exception:
                refused += 1
                try: os.remove(path)
                except: pass

        st.success(f"Ajoutées ✅ : {accepted} | Refusées (qualité) ❌ : {refused}")
        st.rerun()

    photos = planner_get_photos()
    if not photos:
        st.info("Upload des photos 4K (ou baisse le seuil) pour générer une grille.")
    else:
        # analyse + scoring
        items = []
        for p in photos:
            try:
                img = Image.open(p["image_path"])
                feat = compute_features(img)
                sc = score_photo(feat, target)
                items.append({**p, "score": sc, "feat": feat, "tips": advice(feat, target)})
            except Exception:
                pass

        if not items:
            st.warning("Aucune photo analysable.")
        else:
            items_sorted = sorted(items, key=lambda x: x["score"], reverse=True)

            st.markdown("### Top sélection (score)")
            topn = st.slider("Combien de photos à considérer", 9, min(60, len(items_sorted)), min(24, len(items_sorted)))
            pool = items_sorted[:topn]

            if st.button("✨ Générer placement automatique 3×3", type="primary"):
                st.session_state["grid"] = auto_place_grid(pool)

            grid = st.session_state.get("grid")
            if grid:
                st.markdown("## ✅ Grille 3×3 (auto-positionnée)")
                cols = st.columns(3)
                for i, it in enumerate(grid):
                    with cols[i % 3]:
                        st.image(it["image_path"], use_container_width=True)
                        st.caption(f"{it['original_name']} — {it['score']}/100")
                        with st.expander("Conseils retouche"):
                            for t in it["tips"]:
                                st.write("• " + t)

                st.caption("Le placement équilibre clair/sombre et évite de coller les photos trop similaires.")

            st.divider()
            st.markdown("### Toutes les photos (gestion)")
            for it in items_sorted[:60]:
                with st.expander(f"{it['score']}/100 — {it['original_name']} ({it['width']}×{it['height']})"):
                    st.image(make_thumb(Image.open(it["image_path"]), 800), use_container_width=True)
                    for t in it["tips"]:
                        st.write("• " + t)
                    if st.button("🗑️ Supprimer", key=f"del_pl_{it['id']}"):
                        planner_delete(it["id"])
                        st.rerun()

# -------------------------
# PERFORMANCE TAB
# -------------------------
with tab_perf:
    st.subheader("Performance — manuel ou CSV")
    sub1, sub2 = st.tabs(["✍️ Manuel", "⬆️ CSV"])

    with sub1:
        likes_m = st.number_input("Likes moyens", min_value=0, value=120, step=1)
        com_m = st.number_input("Commentaires moyens", min_value=0, value=6, step=1)
        reach_m = st.number_input("Reach moyen", min_value=0, value=1800, step=50)

        eng = (likes_m + 4*com_m) / reach_m if reach_m > 0 else 0.0
        st.metric("Engagement estimé", f"{eng*100:.2f}%")

    with sub2:
        st.caption("CSV : date, likes, comments, reach (optionnel saves, shares)")
        perf = st.file_uploader("Upload performance.csv", type=["csv"])
        if perf:
            dfp = pd.read_csv(perf)
            dfp.columns = [c.strip().lower() for c in dfp.columns]
            needed = {"date","likes","comments","reach"}
            if not needed.issubset(set(dfp.columns)):
                st.error("Colonnes requises : date, likes, comments, reach")
            else:
                dfp["date"] = pd.to_datetime(dfp["date"], errors="coerce")
                dfp = dfp.dropna(subset=["date"]).copy()
                for c in ["likes","comments","reach","saves","shares"]:
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
                st.write("Meilleurs jours")
                st.dataframe(dfp.groupby("weekday")["engagement"].mean().sort_values(ascending=False), use_container_width=True)

# -------------------------
# STORAGE TAB
# -------------------------
with tab_storage:
    st.subheader("Stockage (debug)")

    st.write("### Planner photos")
    ph = planner_get_photos()
    st.write(f"{len(ph)} élément(s)")
    st.write("### Posts")
    st.write(f"{len(get_posts())} post(s)")
    st.write("### Followers DB")
    fdb, gdb = get_follow_lists_from_db()
    st.write(f"followers: {len(fdb)} | following: {len(gdb)}")

    st.caption("⚠️ Streamlit Cloud peut parfois reset le disque lors de redeploy. Pour 100% persistant, il faudrait stockage externe.")
