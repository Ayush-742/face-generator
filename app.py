"""Streamlit demo (monochrome UI).  Run:  python -m streamlit run app.py"""
import io
import random
from pathlib import Path

import streamlit as st
from PIL import ImageOps

from utils import generate, load_generator, noise_from_seed, slerp, to_pil_grid

WEIGHTS = "checkpoints/generator.pth"

st.set_page_config(page_title="FACEGAN", page_icon="◼", layout="centered")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@800;900&family=Space+Mono:wght@400;700&display=swap');

:root { --fg:#fff; --bg:#000; --mid:#8a8a8a; --line:#2b2b2b; }
html, body, .stApp { background: var(--bg) !important; }
.block-container { max-width: 900px; padding-top: 3rem; padding-bottom: 4rem; }

/* hide Streamlit chrome */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
header[data-testid="stHeader"] { background: transparent; }

/* hero */
.kicker { font-family:'Space Mono',ui-monospace,monospace; font-size:.75rem; letter-spacing:.28em;
  text-transform:uppercase; color:var(--mid); margin-bottom:.6rem; }
.title { display:block; font-family:'Archivo','Helvetica Neue',Arial,sans-serif !important; font-weight:900;
  font-size:clamp(3.8rem, 13vw, 7.5rem) !important; line-height:.95 !important; letter-spacing:-.01em;
  text-transform:uppercase; color:var(--fg); margin:0 0 .3rem 0; padding:0; white-space:nowrap; }
.title .outline { color:transparent; -webkit-text-stroke:2px var(--fg); letter-spacing:.03em; margin-left:.05em; }
.tagline { font-family:'Space Mono',ui-monospace,monospace; font-size:.85rem; color:var(--mid);
  margin:1.1rem 0 2rem 0; max-width:34rem; line-height:1.6; }

/* stats strip */
.stats { display:grid; grid-template-columns:repeat(4,1fr); border-top:2px solid var(--fg);
  border-bottom:1px solid var(--line); margin-bottom:2.2rem; }
.stat { padding:.9rem .2rem .9rem 1rem; border-right:1px solid var(--line); }
.stat:last-child { border-right:none; }
.stat .n { font-family:'Archivo','Helvetica Neue',Arial,sans-serif; font-weight:900; font-size:1.7rem;
  letter-spacing:-.03em; line-height:1; }
.stat .l { font-family:'Space Mono',ui-monospace,monospace; font-size:.62rem; letter-spacing:.2em;
  text-transform:uppercase; color:var(--mid); margin-top:.45rem; }
@media (max-width:640px){ .stats{grid-template-columns:repeat(2,1fr);} }

/* tabs */
button[data-baseweb="tab"] { font-family:'Space Mono',ui-monospace,monospace; text-transform:uppercase;
  letter-spacing:.2em; font-weight:700; font-size:.8rem; background:transparent !important; }
[data-baseweb="tab-highlight"] { background:var(--fg) !important; height:3px !important; }
[data-baseweb="tab-border"] { background:var(--line) !important; }

/* labels and captions */
[data-testid="stWidgetLabel"] p { font-family:'Space Mono',ui-monospace,monospace; text-transform:uppercase;
  letter-spacing:.18em; font-size:.68rem !important; color:var(--mid) !important; }
[data-testid="stCaptionContainer"] { font-family:'Space Mono',ui-monospace,monospace; text-transform:uppercase;
  letter-spacing:.14em; font-size:.68rem; color:var(--mid); }

/* buttons: hard edges, invert on hover */
.stButton > button, .stDownloadButton > button {
  background:var(--bg) !important; color:var(--fg) !important; border:2px solid var(--fg) !important;
  border-radius:0 !important; font-family:'Archivo','Helvetica Neue',Arial,sans-serif; font-weight:800;
  text-transform:uppercase; letter-spacing:.14em; padding:.65rem 1.2rem; transition:none; }
.stButton > button p, .stDownloadButton > button p { color:inherit !important; font-weight:800; }
.stButton > button:hover, .stDownloadButton > button:hover { background:var(--fg) !important; color:var(--bg) !important; }

/* inputs: square, white outline */
[data-baseweb="input"], [data-baseweb="base-input"] { border-radius:0 !important; background:var(--bg) !important; }
[data-baseweb="input"] { border:2px solid var(--fg) !important; }
[data-baseweb="base-input"] { border:none !important; }
.stNumberInput button { background:var(--bg) !important; color:var(--fg) !important; border-radius:0 !important; }
input { font-family:'Space Mono',ui-monospace,monospace !important; font-weight:700 !important; }

/* slider + radio text */
[data-testid="stSlider"] [data-testid="stTickBarMin"], [data-testid="stSlider"] [data-testid="stTickBarMax"] { color:var(--mid); }
[data-testid="stRadio"] label p { font-family:'Space Mono',ui-monospace,monospace; text-transform:uppercase;
  letter-spacing:.14em; font-size:.75rem; }

/* faces */
[data-testid="stImage"] img { border:2px solid var(--fg); border-radius:0 !important; }
hr { border-color:var(--line); }
.foot { font-family:'Space Mono',ui-monospace,monospace; font-size:.65rem; letter-spacing:.22em;
  text-transform:uppercase; color:var(--mid); border-top:1px solid var(--line); padding-top:1rem; margin-top:3rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

if not Path(WEIGHTS).exists():
    st.error(f"Missing `{WEIGHTS}`. Train the model first and copy generator.pth there.")
    st.stop()


@st.cache_resource(show_spinner="Loading generator...")
def get_generator():
    return load_generator(WEIGHTS)


g = get_generator()
n_params = sum(p.numel() for p in g.parameters())

st.markdown(
    f"""
<div class="kicker">DCGAN · PyTorch · CelebA</div>
<div class="title">Face<span class="outline">GAN</span></div>
<div class="tagline">Faces that don't exist. Generated from random noise by a neural network trained to fool another one.</div>
<div class="stats">
  <div class="stat"><div class="n">64×64</div><div class="l">Resolution</div></div>
  <div class="stat"><div class="n">50K</div><div class="l">Training images</div></div>
  <div class="stat"><div class="n">30</div><div class="l">Epochs</div></div>
  <div class="stat"><div class="n">{n_params / 1e6:.1f}M</div><div class="l">Parameters</div></div>
</div>
""",
    unsafe_allow_html=True,
)

if "seed" not in st.session_state:
    st.session_state.seed = 42


def randomize():
    st.session_state.seed = random.randint(0, 1_000_000)


def png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


bw = st.radio("Display", ["Color", "Black & white"], horizontal=True) == "Black & white"


def finish(img):
    return ImageOps.grayscale(img) if bw else img


tab1, tab2 = st.tabs(["Random", "Blend"])

with tab1:
    c1, c2 = st.columns([1, 2])
    c1.button("Randomize", on_click=randomize, use_container_width=True)
    seed = c2.number_input("Seed (same seed = same faces)", 0, 10**9, step=1, key="seed")
    n = st.slider("Number of faces", 8, 64, 32, step=8)
    img = finish(to_pil_grid(generate(g, noise_from_seed(seed, n)), nrow=8))
    st.image(img, use_container_width=True)
    st.caption(f"Seed {seed} · {n} faces · none of these people exist")
    st.download_button("Download PNG", png_bytes(img), f"faces_seed{seed}.png", "image/png")

with tab2:
    c1, c2, c3 = st.columns(3)
    a = c1.number_input("Seed A", 0, 10**9, 10, step=1)
    b = c2.number_input("Seed B", 0, 10**9, 500, step=1)
    steps = c3.slider("Steps", 5, 12, 10)
    z = slerp(noise_from_seed(a)[0], noise_from_seed(b)[0], steps)
    img2 = finish(to_pil_grid(generate(g, z), nrow=steps, scale=1))
    st.image(img2, use_container_width=True)
    st.caption("Each frame is a point on a smooth path between face A and face B in latent space")
    st.download_button("Download PNG", png_bytes(img2), f"blend_{a}_{b}.png", "image/png")

st.markdown('<div class="foot">Built with PyTorch · Streamlit</div>', unsafe_allow_html=True)