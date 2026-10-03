import html
import time

import streamlit as st

from inference import supported_words, translate_with_details

st.set_page_config(page_title="Transformer Translate · English to Dutch", page_icon="🌍", layout="wide")

ENGINE_PRE = "Neural engine · any English text"
ENGINE_OWN = "Custom Transformer · limited vocabulary"
EXAMPLES = ["Hello, how are you today?", "Where is the nearest train station?", "I would like a cup of coffee, please.", "I love mathematics"]

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {font-family:'DM Sans',system-ui,sans-serif;}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {display:none !important;}
[data-testid="stHeader"] {background:transparent;}
.block-container {max-width:1080px; padding-top:1.4rem; padding-bottom:3rem;}
.nav {display:flex; justify-content:space-between; align-items:center; padding:6px 0 28px;}
.brand {display:flex; align-items:center; gap:10px; font-weight:700; font-size:19px; letter-spacing:-.01em;}
.logo {width:34px; height:34px; border-radius:9px; background:linear-gradient(135deg,#3547E8,#6a7bff); color:#fff; display:grid; place-items:center; font-size:18px;}
.pill {font-size:13px; color:#5b6485; background:#fff; border:1px solid #d9deec; border-radius:999px; padding:5px 13px;}
.hero h1 {font-size:clamp(32px,5vw,52px); line-height:1.08; letter-spacing:-.03em; margin:0 0 12px; font-weight:700;}
.hero p {color:#5b6485; font-size:18px; max-width:600px; margin:0 0 26px;}
.panel-label {font-size:13px; font-weight:600; color:#5b6485; margin-bottom:8px; display:flex; justify-content:space-between;}
.out-card {background:#fff; border:1px solid #d9deec; border-radius:14px; min-height:220px; padding:18px 20px; font-size:21px; line-height:1.5; font-weight:500; color:#1B2340; box-shadow:0 1px 2px rgba(27,35,64,.04);}
.out-card.empty {color:#9aa3c4; font-weight:400;}
.out-card.warn {color:#9a4a00; background:#fff7ea; border-color:#f3d9ad; font-size:16px; font-weight:400;}
.meta {font-size:13px; color:#7a83a6; margin-top:10px;}
[data-testid="stTextArea"] textarea {font-size:19px; line-height:1.5; border-radius:14px; border:1px solid #d9deec; background:#fff; padding:16px 18px;}
[data-testid="stTextArea"] textarea:focus {border-color:#3547E8; box-shadow:0 0 0 3px rgba(53,71,232,.15);}
[data-testid="stForm"] {border:none; padding:0; background:transparent;}
.stButton > button, [data-testid="stFormSubmitButton"] > button {border-radius:10px; font-weight:600; padding:.55rem 1.1rem;}
div[role="radiogroup"] {gap:10px;}
.foot {color:#7a83a6; font-size:13px; text-align:center; margin-top:42px;}
</style>
<div class="nav">
  <div class="brand"><div class="logo">🌍</div>Transformer Translate</div>
  <span class="pill">English → Dutch</span>
</div>
<div class="hero">
  <h1>Translate English to Dutch, instantly.</h1>
  <p>Type a word, a sentence or a short paragraph. Powered by neural machine translation.</p>
</div>
""",
    unsafe_allow_html=True,
)

engine = st.radio("Engine", [ENGINE_PRE, ENGINE_OWN], horizontal=True, label_visibility="collapsed")


def use_example(text):
    st.session_state["src_text"] = text
    st.session_state["auto"] = True


if engine == ENGINE_PRE:
    chips = EXAMPLES[:3]
else:
    chips = ["i love mathematics", "i love calculus", "calculus is great", "mathematics is useful"]

st.write("")
chip_cols = st.columns(len(chips))
for i, (c, ex) in enumerate(zip(chip_cols, chips)):
    c.button(ex, key=f"chip{i}", use_container_width=True, on_click=use_example, args=(ex,))

st.write("")
left, right = st.columns(2, gap="large")

with left:
    st.markdown('<div class="panel-label"><span>English</span></div>', unsafe_allow_html=True)
    with st.form("translate_form", clear_on_submit=False):
        text = st.text_area(
            "English text",
            key="src_text",
            height=220,
            max_chars=1000,
            placeholder="Type or paste English text here…",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Translate", type="primary", use_container_width=True)

if "history" not in st.session_state:
    st.session_state["history"] = []

if (submitted or st.session_state.pop("auto", False)) and text.strip():
    start = time.perf_counter()
    result = {"engine": engine, "source": text.strip()}
    try:
        if engine == ENGINE_PRE:
            from pretrained import translate_pretrained

            with st.spinner("Translating…"):
                result["output"] = translate_pretrained(text.strip())
        else:
            r = translate_with_details(text.strip())
            if r["unknown"]:
                result["warning"] = (
                    "This engine only knows a small vocabulary and doesn't recognise: "
                    + ", ".join(r["unknown"])
                    + ". Switch to the neural engine for general text."
                )
            else:
                result["output"] = r["translation"]
    except ImportError:
        result["warning"] = "Required packages are missing. Run: python -m pip install -r requirements.txt"
    except Exception as e:  # network / model loading problems
        result["warning"] = f"The translation engine could not run: {e}"
    result["ms"] = int((time.perf_counter() - start) * 1000)
    st.session_state["result"] = result
    if "output" in result:
        st.session_state["history"].insert(0, (result["source"], result["output"]))
        st.session_state["history"] = st.session_state["history"][:5]
elif submitted:
    st.session_state["result"] = {"warning": "Please enter some text to translate."}

with right:
    st.markdown('<div class="panel-label"><span>Dutch</span></div>', unsafe_allow_html=True)
    res = st.session_state.get("result")
    if not res:
        st.markdown('<div class="out-card empty">Your translation will appear here.</div>', unsafe_allow_html=True)
    elif "warning" in res:
        st.markdown(f'<div class="out-card warn">{html.escape(res["warning"])}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="out-card">{html.escape(res["output"])}</div>', unsafe_allow_html=True)
        name = "Neural engine" if res["engine"] == ENGINE_PRE else "Custom Transformer"
        st.markdown(f'<div class="meta">{name} · {res["ms"]} ms · {len(res["source"])} characters</div>', unsafe_allow_html=True)
        with st.expander("Copy text"):
            st.code(res["output"], language=None)

if st.session_state["history"]:
    with st.expander("Recent translations"):
        for src, out in st.session_state["history"]:
            st.markdown(f"**{html.escape(src)}**  \n{html.escape(out)}")

with st.expander("About this project"):
    st.write(
        "**Neural engine:** Helsinki-NLP `opus-mt-en-nl` (MarianMT), a pretrained Transformer for English to Dutch.\n\n"
        "**Custom Transformer:** an encoder-decoder Transformer implemented from scratch in PyTorch "
        "(multi-head attention, positional encoding, causal masking, autoregressive decoding). "
        "It uses a deliberately small vocabulary: " + ", ".join(f"`{w}`" for w in supported_words()) + "."
    )

st.markdown('<div class="foot">Built by Anto Jovita · PyTorch · Streamlit</div>', unsafe_allow_html=True)
