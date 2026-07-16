import os
import logging
import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Load .env for local dev; on Streamlit Cloud, st.secrets is used instead (see src/query.py)
from dotenv import load_dotenv
load_dotenv()

# Push Streamlit secrets into env too, so any os.getenv() call downstream still works
if "gemini_token" in st.secrets:
    os.environ["gemini_token"] = st.secrets["gemini_token"]

from src.retrieval import load_data, build_bm25_index, hybrid_retrieve, rerank_chunks
from src.prompt import build_prompt
from src.memory import build_history_context
from src.analyzer import analyze_query, detect_and_translate, translate_response, translate_bot_message
from src.query import query_gem
from src.emergency import check_emergency, EMERGENCY_RESPONSE

st.set_page_config(page_title="Sakhii Bot", page_icon="🩺", layout="centered")

# ── Tighten layout / deepen contrast ──────────────────────────────
st.markdown("""
<style>
    /* Reduce big empty top gap */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 780px;
    }

    /* Compact the title */
    h1 {
        font-size: 1.9rem !important;
        margin-bottom: 0.2rem !important;
    }

    /* Tighter caption under title */
    [data-testid="stCaptionContainer"] {
        margin-bottom: 0.6rem !important;
    }

    /* Disclaimer info box: less padding, darker text for contrast */
    div[data-testid="stAlert"] {
        padding: 0.6rem 0.9rem !important;
        margin-bottom: 0.8rem !important;
    }
    div[data-testid="stAlert"] p {
        color: #3a2430 !important;
        font-size: 0.9rem !important;
    }

    /* Sidebar: deepen background, darken labels for contrast */
    section[data-testid="stSidebar"] {
        background-color: #FFD1CE !important;
    }
    section[data-testid="stSidebar"] label {
        color: #5A1030 !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] h2 {
        color: #7A0038 !important;
    }

    /* Buttons: solid, higher-contrast magenta */
    button[kind="secondary"], .stButton button {
        border: 1.5px solid #BE0056 !important;
        color: #BE0056 !important;
        font-weight: 600 !important;
    }
    .stButton button:hover {
        background-color: #BE0056 !important;
        color: white !important;
    }

    /* Chat messages: reduce vertical gaps between turns */
    div[data-testid="stChatMessage"] {
        padding-top: 0.35rem !important;
        padding-bottom: 0.35rem !important;
    }

    /* Chat input: darker placeholder text for contrast */
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #8a5560 !important;
        opacity: 1 !important;
    }
</style>
""", unsafe_allow_html=True)

DATA_PATH = "data/medical_kb.txt"


# ── Cached one-time setup ─────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base and building index... (first load can take a minute)")
def initialize():
    chunks = load_data(DATA_PATH)
    build_bm25_index(chunks)
    return len(chunks)


# ── Session state ──────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []          # [{"user": ..., "bot": ...}]
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {"preferred_language": "english"}
if "profile_done" not in st.session_state:
    st.session_state.profile_done = False


def rag_answer(query: str) -> str:
    profile = st.session_state.user_profile
    preferred_lang = profile.get("preferred_language", "english")

    if check_emergency(query):
        logger.warning(f"Emergency detected: '{query}'")
        return EMERGENCY_RESPONSE

    lang_info = detect_and_translate(query, preferred_language=preferred_lang)
    english_query = lang_info["english_query"]

    analysis = analyze_query(english_query, profile)
    intent = analysis["intent"]
    symptoms = analysis["symptoms"]
    rewritten = analysis["rewritten_query"]

    history_context = build_history_context(st.session_state.chat_history)

    try:
        if intent == "casual":
            name = profile.get("name", "")
            greeting = f"Hi {name}! " if name else "Hi! "
            answer = (
                f"{greeting}I'm Sakhii, your personal health assistant "
                f"created by Sakhii Care Foundation. "
                f"I can help with symptoms, illnesses, medications, "
                f"and general health advice. What's on your mind?"
            )
            answer = translate_bot_message(answer, preferred_lang)

        elif intent == "irrelevant":
            answer = "I can only answer health related questions."
            answer = translate_bot_message(answer, preferred_lang)

        else:  # health
            retrieved, mode = hybrid_retrieve(rewritten, n_results=5)
            sources = []
            if mode == "chunks" and retrieved:
                retrieved, sources = rerank_chunks(rewritten, retrieved, top_n=3)

            prompt = build_prompt(
                query=english_query,
                retrieved_chunks=retrieved,
                mode=mode,
                user_profile=profile,
                history_context=history_context,
                symptoms=symptoms,
                sources=sources
            )

            answer = query_gem(prompt)
            answer = translate_response(answer, preferred_lang)

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        answer = "Sorry, something went wrong. Please try again."
        answer = translate_bot_message(answer, preferred_lang)

    return answer


# ── Sidebar: profile ────────────────────────────────────────────────
with st.sidebar:
    st.header("🌐 Your Profile")
    st.caption("All fields optional — helps personalize answers.")

    lang_choice = st.radio(
        "Preferred language",
        options=["english", "hinglish"],
        format_func=lambda x: "English" if x == "english" else "Hinglish (Hindi + English)",
        index=0 if st.session_state.user_profile.get("preferred_language", "english") == "english" else 1,
    )

    name = st.text_input("What should I call you?", value=st.session_state.user_profile.get("name", ""))
    age = st.text_input("Age", value=str(st.session_state.user_profile.get("age", "")))
    gender = st.text_input("Gender", value=st.session_state.user_profile.get("gender", ""))
    conditions = st.text_input(
        "Existing conditions (comma separated)",
        value=", ".join(st.session_state.user_profile.get("conditions", []))
    )
    medications = st.text_input(
        "Current medications (comma separated)",
        value=", ".join(st.session_state.user_profile.get("medications", []))
    )

    if st.button("Save profile", use_container_width=True):
        profile = {"preferred_language": lang_choice}
        if name.strip():
            profile["name"] = name.strip()
        if age.strip():
            try:
                profile["age"] = int(age.strip())
            except ValueError:
                profile["age"] = age.strip()
        if gender.strip():
            profile["gender"] = gender.strip()
        if conditions.strip():
            profile["conditions"] = [c.strip() for c in conditions.split(",") if c.strip()]
        if medications.strip():
            profile["medications"] = [m.strip() for m in medications.split(",") if m.strip()]
        st.session_state.user_profile = profile
        st.session_state.profile_done = True
        st.success("Profile saved!")

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.caption(
        "⚠️ This assistant is for general health information only "
        "and is not a substitute for professional medical advice."
    )


# ── Main chat UI ────────────────────────────────────────────────────
st.title("🩺 Sakhii Bot")
st.caption("Your personal health assistant — Sakhii Care Foundation")

n_chunks = initialize()

for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(turn["user"])
    with st.chat_message("assistant"):
        st.write(turn["bot"])

# Fill empty state with example questions instead of blank space
if not st.session_state.chat_history:
    st.markdown("**Try asking:**")
    examples = [
        "What causes frequent headaches?",
        "How can I manage period cramps?",
        "What are early signs of anemia?",
        "Tips for better sleep",
    ]
    cols = st.columns(2)
    example_clicked = None
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, use_container_width=True, key=f"ex_{i}"):
            example_clicked = ex
else:
    example_clicked = None

query = st.chat_input("Ask a health question...")
if example_clicked:
    query = example_clicked
if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = rag_answer(query)
        st.write(answer)

    st.session_state.chat_history.append({"user": query, "bot": answer})
