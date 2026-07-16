import os
import logging
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# ── Make Streamlit Cloud secrets available as env vars ──────────
# query.py reads os.getenv("gemini_token"). Locally this comes from .env.
# On Streamlit Community Cloud, secrets are set in the dashboard and
# exposed via st.secrets, so we mirror them into os.environ here.
if "gemini_token" in st.secrets:
    os.environ["gemini_token"] = st.secrets["gemini_token"]

from src.retrieval import load_data, build_bm25_index, hybrid_retrieve, rerank_chunks
from src.prompt import build_prompt
from src.memory import build_history_context
from src.analyzer import analyze_query, detect_and_translate, translate_response, translate_bot_message
from src.query import query_gem
from src.emergency import check_emergency, EMERGENCY_RESPONSE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Sakhii — Health Assistant", page_icon="🩺", layout="centered")


# ── Load KB + build BM25 index once per deployment, not per user ─
@st.cache_resource(show_spinner="Loading knowledge base…")
def initialize():
    chunks = load_data("data/medical_kb.txt")
    build_bm25_index(chunks)
    return True


initialize()

# ── Per-user session state ───────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "profile" not in st.session_state:
    st.session_state.profile = {"preferred_language": "english"}


def rag_answer(query: str) -> str:
    profile = st.session_state.profile
    preferred_lang = profile.get("preferred_language", "english")

    # Step 0 — Emergency check
    if check_emergency(query):
        logger.warning(f"Emergency detected: '{query}'")
        st.session_state.chat_history.append({"user": query, "bot": EMERGENCY_RESPONSE})
        return EMERGENCY_RESPONSE

    # Step 1 — Language detection + translation
    lang_info = detect_and_translate(query, preferred_language=preferred_lang)
    english_query = lang_info["english_query"]

    # Step 2 — Analyze query
    analysis = analyze_query(english_query, profile)
    intent = analysis["intent"]
    symptoms = analysis["symptoms"]
    rewritten = analysis["rewritten_query"]

    # Step 3 — History context
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

    st.session_state.chat_history.append({"user": query, "bot": answer})
    return answer


# ── Sidebar: profile ──────────────────────────────────────────────
with st.sidebar:
    st.header("Your profile")
    st.caption("All fields optional — personalizes your answers.")

    lang_choice = st.selectbox("Preferred language", ["English", "Hinglish"])
    name = st.text_input("Name")
    age = st.text_input("Age")
    gender = st.text_input("Gender")
    conditions = st.text_input("Existing conditions (comma separated)")
    medications = st.text_input("Current medications (comma separated)")

    if st.button("Save profile", use_container_width=True):
        profile = {"preferred_language": lang_choice.lower()}
        if name:
            profile["name"] = name
        if age:
            try:
                profile["age"] = int(age)
            except ValueError:
                profile["age"] = age
        if gender:
            profile["gender"] = gender
        if conditions:
            profile["conditions"] = [c.strip() for c in conditions.split(",") if c.strip()]
        if medications:
            profile["medications"] = [m.strip() for m in medications.split(",") if m.strip()]
        st.session_state.profile = profile
        st.success("Profile saved!")

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ── Main chat UI ───────────────────────────────────────────────────
st.title("🩺 Sakhii — Your Personal Health Assistant")
st.caption("Developed for Sakhii Care Foundation")
st.info(
    "Sakhii provides general health information and is not a substitute "
    "for professional medical advice. In an emergency, contact local "
    "emergency services immediately.",
    icon="ℹ️",
)

for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(turn["user"])
    with st.chat_message("assistant"):
        st.write(turn["bot"])

query = st.chat_input("Type your health question…")
if query:
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer = rag_answer(query)
        st.write(answer)
