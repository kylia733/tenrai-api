"""
Deteksi jari -> angka (MAL ID) -> Tenrai API (data anime) di Streamlit.

Gestur:
  OK (jempol + telunjuk bertemu)  = mulai input / selesai dan cari
  Angka 1-9 (dua tangan dijumlah) = satu digit
  Kepalan                         = digit 0 (hanya di mode INPUT)
  Dua tangan terbuka (10)         = reset (batal + lepas kunci gestur)

Jalankan:  uv run streamlit run main.py
"""
import html
import math
import time
from collections import Counter, deque
from pathlib import Path
from threading import Lock
import av
import cv2
import mediapipe as mp
import numpy as np
import requests
import streamlit as st
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import webrtc_streamer

# ---------- Pengaturan (silakan diutak-atik) ----------
HOLD_DIGIT = 0.8      # detik menahan angka supaya dihitung 1 digit
HOLD_OK = 1.0         # detik menahan gestur OK supaya dianggap perintah
HOLD_RESET = 0.8      # detik menahan dua tangan terbuka untuk reset
OK_PINCH = 0.35       # makin besar = gestur OK makin gampang terdeteksi
INPUT_TIMEOUT = 30    # detik tanpa aktivitas di mode INPUT -> batal
MAX_DIGITS = 6        # maksimal digit 
VIDEO_HEIGHT = 480    # px, tinggi video kamera (tetap, apa pun lebar layar)
PANEL_HEIGHT = VIDEO_HEIGHT + 50  # px, tinggi panel anime = video + bar STOP; ubah 50 kalau kurang pas
SMOOTH_FRAMES = 7     # voting mayoritas dari N frame terakhir

API_BASE = "https://api.tenrai.org/v1"  # Tenrai, pengganti Jikan (skema sama)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

# Garis penghubung 21 titik tangan (untuk digambar manual)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def ensure_model():
    """Unduh model hand landmarker sekali saja kalau belum ada."""
    if not MODEL_PATH.exists():
        r = requests.get(MODEL_URL, timeout=60)
        r.raise_for_status()
        MODEL_PATH.write_bytes(r.content)


# ---------- Hitung jari ----------
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def count_fingers(lm):
    """Hitung jari terangkat di satu tangan (tidak peduli kiri/kanan atau rotasi)."""
    wrist = lm[0]
    count = 0
    # Telunjuk, tengah, manis, kelingking: ujung lebih jauh dari pergelangan
    # dibanding sendi tengahnya -> jari lurus.
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        if dist(lm[tip], wrist) > dist(lm[pip], wrist):
            count += 1
    # Jempol: ujung menjauh dari pangkal telunjuk dibanding sendi di bawahnya.
    # Kalau jempol sering salah baca, ubah angka 1.2 ini.
    if dist(lm[4], lm[5]) > dist(lm[3], lm[5]) * 1.2:
        count += 1
    return count


def is_ok(lm):
    """Gestur OK: ujung jempol menempel ke ujung telunjuk, jari lain lurus."""
    palm = dist(lm[0], lm[9])  # pergelangan -> pangkal jari tengah (ukuran tangan)
    pinch = dist(lm[4], lm[8]) < palm * OK_PINCH
    straight = sum(
        dist(lm[tip], lm[0]) > dist(lm[pip], lm[0])
        for tip, pip in ((12, 10), (16, 14), (20, 18))
    )
    return pinch and straight >= 2


# ---------- State machine ----------
class Tracker:
    def __init__(self):
        self.lock = Lock()
        ensure_model()
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.last_ts = 0
        self.state = "IDLE"
        self.buffer = []
        self.history = deque(maxlen=SMOOTH_FRAMES)
        self.candidate = None
        self.since = time.time()
        self.locked = None          # nilai yang baru dipakai; harus berubah dulu
        self.last_activity = time.time()
        self.result = None          # angka final, diambil oleh thread Streamlit
        self.clear = False          # True = minta panel anime dikosongkan

    def read(self, rgb):
        """Return (pembacaan, daftar landmark tiap tangan). Pembacaan: 1-9, 'FIST', 'OK', 'RESET', atau None."""
        # Mode VIDEO butuh timestamp (ms) yang selalu naik
        ts = max(int(time.time() * 1000), self.last_ts + 1)
        self.last_ts = ts
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.landmarker.detect_for_video(mp_image, ts)
        hands = res.hand_landmarks
        if not hands:
            return None, []
        if any(is_ok(lm) for lm in hands):
            return "OK", hands           # dicek duluan, sebelum hitung jari
        total = sum(count_fingers(lm) for lm in hands)
        if total == 0:
            return "FIST", hands         # semua tangan yang terlihat mengepal
        if 1 <= total <= 9:
            return total, hands
        if total == 10:
            return "RESET", hands        # dua tangan terbuka
        return None, hands

    def update(self, reading, now):
        self.history.append(reading)
        smooth = Counter(self.history).most_common(1)[0][0]

        if smooth != self.candidate:
            self.candidate = smooth
            self.since = now
        if self.locked is not None and smooth != self.locked:
            self.locked = None

        progress = 0.0
        # Di IDLE hanya OK dan RESET yang dianggap; angka & kepalan hanya di INPUT
        eligible = smooth is not None and smooth != self.locked and (
            self.state == "INPUT" or smooth in ("OK", "RESET")
        )
        if eligible:
            need = {"OK": HOLD_OK, "RESET": HOLD_RESET}.get(smooth, HOLD_DIGIT)
            progress = min((now - self.since) / need, 1.0)
            if progress >= 1.0:
                self.commit(smooth, now)
                progress = 0.0

        if self.state == "INPUT" and now - self.last_activity > INPUT_TIMEOUT:
            self.state = "IDLE"
            self.buffer = []

        return smooth, progress

    def commit(self, value, now):
        self.locked = value
        self.last_activity = now
        if value == "RESET":
            self.state = "IDLE"
            self.buffer = []
            self.clear = True
        elif value == "OK":
            if self.state == "IDLE":
                self.state = "INPUT"
                self.buffer = []
            else:
                if self.buffer:
                    self.result = int("".join(map(str, self.buffer)))
                self.state = "IDLE"
                self.buffer = []
        elif self.state == "INPUT" and len(self.buffer) < MAX_DIGITS:
            digit = 0 if value == "FIST" else value
            if digit == 0 and not self.buffer:
                return  # ID tidak diawali 0
            self.buffer.append(digit)

    def snapshot(self):
        with self.lock:
            result, self.result = self.result, None
            clear, self.clear = self.clear, False
            return {
                "clear": clear,
                "state": self.state,
                "buffer": "".join(map(str, self.buffer)),
                "result": result,
            }


@st.cache_resource
def get_tracker():
    return Tracker()


tracker = get_tracker()


# ---------- Tampilan di dalam kamera (hitam putih, font Times New Roman) ----------
def load_font(size):
    """Cari Times New Roman; kalau tidak ada, pakai font serif lain."""
    for name in ("times.ttf", "Times New Roman.ttf", "LiberationSerif-Regular.ttf", "DejaVuSerif.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


STATE_LABEL = {"IDLE": "Menunggu", "INPUT": "Input"}
_FONTS = {}


def get_fonts(w):
    """Ukuran font mengikuti lebar frame supaya proporsional."""
    size = max(14, w // 34)
    if size not in _FONTS:
        _FONTS[size] = (load_font(size), load_font(max(12, int(size * 0.75))), size)
    return _FONTS[size]


def draw_overlay(img, state, label, buffer, progress):
    h, w = img.shape[:2]
    big, small, size = get_fonts(w)
    lh = int(size * 1.45)
    pad = int(size * 0.6)
    panel_w, panel_h = int(size * 10), lh * 3 + pad
    band = int(size * 3.0)

    # Panel hitam semi-transparan: kiri atas (info) dan bawah (petunjuk + bar)
    layer = img.copy()
    cv2.rectangle(layer, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.rectangle(layer, (0, h - band), (w, h), (0, 0, 0), -1)
    img = cv2.addWeighted(layer, 0.65, img, 0.35, 0)

    # Bar progres putih
    bar_h = max(8, size // 2)
    x0, x1 = pad, w - pad
    y1 = h - pad
    y0 = y1 - bar_h
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 1)
    if progress > 0:
        cv2.rectangle(img, (x0, y0), (x0 + int((x1 - x0) * progress), y1), (255, 255, 255), -1)

    # Teks pakai PIL supaya bisa memakai font TTF (Times New Roman)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    white = (255, 255, 255)
    rows = (
        ("Mode", STATE_LABEL.get(state, state)),
        ("Jari", label),
        ("Angka", buffer or "-"),
    )
    for i, (k, v) in enumerate(rows):
        y = pad // 2 + i * lh
        d.text((pad, y), k, font=big, fill=white)
        d.text((pad + int(size * 3.8), y), f":  {v}", font=big, fill=white)
    hint = (
        "Buat gestur OK selama 1 detik untuk memulai."
        if state == "IDLE"
        else "Angka (kepalan = 0). OK untuk selesai. Dua tangan terbuka untuk reset."
    )
    d.text((pad, h - band + pad // 2), hint, font=small, fill=white)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ---------- Callback video (jalan di thread terpisah, jangan pakai st.*) ----------
def video_callback(frame):
    img = cv2.flip(frame.to_ndarray(format="bgr24"), 1)  # mirror biar natural
    with tracker.lock:
        reading, hands = tracker.read(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        smooth, progress = tracker.update(reading, time.time())
        state = tracker.state
        buffer = "".join(map(str, tracker.buffer))

    h_img, w_img = img.shape[:2]
    for lm in hands:
        pts = [(int(p.x * w_img), int(p.y * h_img)) for p in lm]
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], (255, 255, 255), 2)
        for pt in pts:
            cv2.circle(img, pt, 5, (0, 0, 0), -1)
            cv2.circle(img, pt, 3, (255, 255, 255), -1)

    label = {None: "-", "FIST": "0", "OK": "OK", "RESET": "Reset"}.get(smooth, str(smooth))
    img = draw_overlay(img, state, label, buffer, progress)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------- Jikan API ----------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_rank(rank: int):
    """Ambil anime berdasarkan MAL ID (21 = One Piece). Pakai Tenrai API. Error tidak ikut di-cache."""
    try:
        r = requests.get(f"{API_BASE}/anime/{rank}", timeout=10)
    except requests.RequestException as e:
        raise RuntimeError(f"Gagal menghubungi Tenrai: {e}")
    if r.status_code == 404:
        raise RuntimeError(f"Anime dengan ID {rank} tidak ditemukan.")
    if r.status_code == 429:
        raise RuntimeError("Kena rate limit API, tunggu sebentar lalu coba lagi.")
    if r.status_code != 200:
        raise RuntimeError(f"Tenrai error (HTTP {r.status_code}).")
    j = r.json()
    return j.get("data", j)  # skema Jikan v4: data ada di kunci "data"


def show_anime(placeholder, rank, a):
    with placeholder.container(height=PANEL_HEIGHT, border=True):
        title = a.get("title_english") or a["title"]
        left, right = st.columns([1, 1.9])
        with left:
            st.image(a["images"]["jpg"]["large_image_url"], width=300)
        with right:
            st.markdown(f"## {title}")
            if a.get("title_japanese"):
                st.markdown(f"*{html.escape(a['title_japanese'])}*")
            m1, m2, m3 = st.columns(3)
            m1.metric("Skor", a.get("score") or "-")
            m2.metric("Peringkat", f"#{a['rank']}" if a.get("rank") else "-")
            m3.metric("Episode", a.get("episodes") or "?")
            genres = ", ".join(g["name"] for g in a.get("genres", [])) or "-"
            st.markdown(
                f"**ID:** {rank} &nbsp;|&nbsp; **Tahun:** {a.get('year') or '-'} "
                f"&nbsp;|&nbsp; **Status:** {a.get('status') or '-'}  \n**Genre:** {genres}"
            )
            synopsis = html.escape((a.get("synopsis") or "Tidak ada sinopsis.")[:350] + "...")
            st.markdown(
                f"<div style='font-size:1rem;line-height:1.5'>{synopsis}</div>",
                unsafe_allow_html=True,
            )
            st.link_button("Buka di MyAnimeList", a["url"])


def show_empty(placeholder, text="Belum ada anime. Bentuk angka ID dengan jari."):
    with placeholder.container(height=PANEL_HEIGHT, border=True):
        st.markdown(f"<div style='font-size:1.1rem'>{html.escape(text)}</div>", unsafe_allow_html=True)


def status_box(placeholder, text):
    placeholder.markdown(f'<div class="box">{html.escape(text)}</div>', unsafe_allow_html=True)


# ---------- UI Streamlit ----------
st.set_page_config(page_title="Projek67", layout="wide")
st.markdown(
    """
    <style>
    html, body, .stApp, h1, h2, h3, h4, p, label, .box,
    [data-testid="stMarkdownContainer"] *, [data-testid="stCaptionContainer"] *,
    [data-testid="stMetric"] *, a[data-testid="stBaseLinkButton-secondary"] * {
        font-family: "Times New Roman", Times, serif !important;
    }
    .stApp { background: #ffffff; color: #000000; font-size: 1.05rem; }
    h1, h2, h3, p, span, label, .stMarkdown { color: #000000 !important; }
    div[data-testid="stAlert"] {
        background: #ffffff; color: #000000;
        border: 1px solid #000000; border-radius: 0;
    }
    div[data-testid="stAlert"] * { color: #000000 !important; }
    a[data-testid="stBaseLinkButton-secondary"], button[kind="secondary"] {
        background: #000000; color: #ffffff !important; border: 1px solid #000000; border-radius: 0;
    }
    a[data-testid="stBaseLinkButton-secondary"] * { color: #ffffff !important; }
    img { border: 1px solid #000000; }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-color: #000000 !important; border-radius: 0 !important; }
    .box { border: 1px solid #000000; padding: 14px 18px; font-size: 1.2rem; margin: 1.2rem 0 0 0; }
    </style>
    """,
    unsafe_allow_html=True,
)
head_l, head_r = st.columns([4, 5])
with head_l:
    st.title("Projek67")
    st.caption("OK 1 detik = mulai -> tunjukkan angka 0-9 (kepalan = 0, dua tangan dijumlah) -> OK lagi = cari. Dua tangan terbuka = reset.")
status_ph = head_r.empty()

col1, col2 = st.columns([4, 5])
with col1:
    ctx = webrtc_streamer(
        key="hand-tracker",
        video_frame_callback=video_callback,
        media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
        video_html_attrs={
            "style": {
                "width": "100%",
                "height": f"{VIDEO_HEIGHT}px",
                "objectFit": "contain",
                "backgroundColor": "#000000",
            },
            "controls": True,
            "autoPlay": True,
            "muted": True,
        },
    )

anime_ph = col2.empty()

if "last" in st.session_state:
    show_anime(anime_ph, *st.session_state.last)
else:
    show_empty(anime_ph)

while ctx.state.playing:
    snap = tracker.snapshot()

    if snap["clear"]:
        st.session_state.pop("last", None)
        show_empty(anime_ph)

    if snap["result"] is not None:
        rank = snap["result"]
        try:
            with st.spinner(f"Mencari anime ID {rank}..."):
                anime = fetch_rank(rank)
            st.session_state.last = (rank, anime)
            show_anime(anime_ph, rank, anime)
        except RuntimeError as e:
            show_empty(anime_ph, str(e))

    if snap["state"] == "IDLE":
        status_box(status_ph, "Menunggu. Buat gestur OK 1 detik untuk mulai.")
    else:
        status_box(status_ph, f"Mode INPUT. Angka: {snap['buffer'] or '-'}. OK untuk selesai, dua tangan terbuka untuk reset.")

    time.sleep(0.2)