import pickle
import re
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "trained_model.pkl"
VECTORIZER_CANDIDATES = [
    "vectorizer.pkl",
    "vectorizzed.pkl",
    "tfidf_vectorizer.pkl",
    "tfidf.pkl",
    "vectorizer.sav",
]

LABELS = {
    0: "Offensive",
    1: "Not Offensive",
}

LABEL_COLORS = {
    0: {"bg": "#3d1a1a", "accent": "#f87171", "emoji": "⚠️"},
    1: {"bg": "#0d3d2e", "accent": "#34d399", "emoji": "✅"},
}

SAMPLE_TWEETS = [
    "I love spending time with my friends!",
    "You are the worst person I have ever met",
    "What a beautiful day to learn something new",
]

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 820px;
    }

    #MainMenu, footer, header { visibility: hidden; }

    .hero {
        background: linear-gradient(135deg, #1d9bf0 0%, #7856ff 55%, #f91880 100%);
        border-radius: 20px;
        padding: 2.2rem 2rem;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 18px 40px rgba(29, 155, 240, 0.25);
    }

    .hero h1 {
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.03em;
    }

    .hero p {
        margin: 0;
        opacity: 0.92;
        font-size: 1.02rem;
    }

    .card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1rem;
    }

    .card-title {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 0.75rem;
    }

    .result-card {
        border-radius: 18px;
        padding: 1.6rem;
        margin-top: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .result-label {
        font-size: 1.55rem;
        font-weight: 800;
        margin: 0.2rem 0 0.8rem 0;
        letter-spacing: -0.02em;
    }

    .confidence-track {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 999px;
        height: 12px;
        overflow: hidden;
        margin: 0.6rem 0 0.2rem 0;
    }

    .confidence-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.4s ease;
    }

    .stat-row {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
    }

    .stat-box {
        flex: 1;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    .stat-box span {
        display: block;
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
    }

    .stat-box strong {
        font-size: 1.35rem;
        font-weight: 700;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    div[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    .sidebar-badge {
        display: inline-block;
        background: rgba(29, 155, 240, 0.15);
        color: #7dd3fc;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    div[data-testid="stTextArea"] textarea {
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        background: rgba(255, 255, 255, 0.03) !important;
        font-size: 1rem !important;
        line-height: 1.5 !important;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: #1d9bf0 !important;
        box-shadow: 0 0 0 3px rgba(29, 155, 240, 0.2) !important;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #1d9bf0, #6366f1) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 1.2rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 8px 24px rgba(29, 155, 240, 0.35) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 12px 28px rgba(29, 155, 240, 0.45) !important;
    }

    div[data-testid="stButton"] button:not([kind="primary"]) {
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        background: rgba(255, 255, 255, 0.04) !important;
        font-size: 0.85rem !important;
    }
</style>
"""


def clean_tweet(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


def find_vectorizer_path() -> Path | None:
    for name in VECTORIZER_CANDIDATES:
        path = BASE_DIR / name
        if path.exists():
            return path
    return None


def vectorizer_cache_key() -> str:
    path = find_vectorizer_path()
    if path is None:
        return "missing"
    return f"{path.name}:{path.stat().st_mtime_ns}"


@st.cache_resource
def load_vectorizer(_cache_key: str):
    vectorizer_path = find_vectorizer_path()
    if vectorizer_path is None:
        return None
    with open(vectorizer_path, "rb") as file:
        return pickle.load(file)


def predict(text: str, model, vectorizer):
    cleaned = clean_tweet(text)
    if not cleaned:
        raise ValueError("Please enter some text after cleaning.")

    features = vectorizer.transform([cleaned])
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    confidence = float(probabilities[prediction]) * 100
    return prediction, confidence, cleaned, probabilities


def render_result(prediction: int, confidence: float, probabilities, cleaned: str):
    style = LABEL_COLORS.get(prediction, LABEL_COLORS[0])
    label = LABELS.get(prediction, f"Class {prediction}")
    prob_0 = float(probabilities[0]) * 100
    prob_1 = float(probabilities[1]) * 100

    st.markdown(
        f"""
        <div class="result-card" style="background: {style['bg']};">
            <div class="card-title" style="color: {style['accent']};">Prediction Result</div>
            <div class="result-label" style="color: {style['accent']};">
                {style['emoji']} {label}
            </div>
            <div style="color: #cbd5e1; font-size: 0.9rem;">Model confidence</div>
            <div class="confidence-track">
                <div class="confidence-fill" style="width: {confidence:.1f}%; background: {style['accent']};"></div>
            </div>
            <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.35rem;">{confidence:.1f}% confident</div>
            <div class="stat-row">
                <div class="stat-box">
                    <span>Offensive</span>
                    <strong style="color: #f87171;">{prob_0:.1f}%</strong>
                </div>
                <div class="stat-box">
                    <span>Not Offensive</span>
                    <strong style="color: #34d399;">{prob_1:.1f}%</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🔍 Cleaned text used for prediction"):
        st.code(cleaned, language=None)


st.set_page_config(
    page_title="TweetSense — ML Classifier",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<span class="sidebar-badge">ML Powered</span>', unsafe_allow_html=True)
    st.markdown("### 🐦 TweetSense")
    st.markdown(
        "Classify tweets as **offensive** or **not offensive** "
        "using your trained Logistic Regression model."
    )
    st.divider()
    st.markdown("**Model details**")
    vectorizer_path = find_vectorizer_path()
    if vectorizer_path:
        st.markdown(f"📦 Vectorizer: `{vectorizer_path.name}`")
    st.markdown("🧠 Algorithm: Logistic Regression")
    st.markdown("🏷️ Classes: 0 · 1")
    st.divider()
    st.markdown("**Try a sample**")
    for i, sample in enumerate(SAMPLE_TWEETS):
        if st.button(sample, key=f"sample_{i}", use_container_width=True):
            st.session_state.tweet_input = sample
            st.rerun()

vectorizer = load_vectorizer(vectorizer_cache_key())

st.markdown(
    """
    <div class="hero">
        <h1>🐦 TweetSense</h1>
        <p>Instant tweet classification powered by machine learning</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if vectorizer is None:
    st.error(
        "Missing `vectorizer.pkl`. Save your TF-IDF vectorizer next to `trained_model.pkl`, "
        "then refresh the page."
    )
    st.stop()

model = load_model()

st.markdown('<div class="card"><div class="card-title">Compose Tweet</div></div>', unsafe_allow_html=True)

if "tweet_input" not in st.session_state:
    st.session_state.tweet_input = ""

user_text = st.text_area(
    "Enter tweet text",
    placeholder="What's happening? Type or paste a tweet here...",
    height=130,
    label_visibility="collapsed",
    key="tweet_input",
)

col1, col2 = st.columns([3, 1])
with col1:
    predict_clicked = st.button("✨ Analyze Tweet", type="primary", use_container_width=True)
with col2:
    if st.button("Clear", use_container_width=True):
        st.session_state.tweet_input = ""
        st.rerun()

if predict_clicked:
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        try:
            prediction, confidence, cleaned, probabilities = predict(
                user_text, model, vectorizer
            )
            render_result(prediction, confidence, probabilities, cleaned)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
