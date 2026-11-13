"""FaceGAN demo (light editorial UI).  Run:  python -m streamlit run app.py"""
import base64
import csv
import io
import random
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from utils import generate, load_generator, noise_from_seed, slerp, to_pil_grid

HERE = Path(__file__).parent
WEIGHTS = HERE / "checkpoints" / "generator.pth"
ASSETS = HERE / "assets"

st.set_page_config(page_title="FaceGAN", page_icon="◐", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter+Tight:ital,wght@0,300;0,400;0,500;0,600;1,400;1,500&display=swap');
:root { --ink:#0e0e0e; --mid:#7b7b7b; --soft:#f4f4f4; --soft2:#ebebeb; --line:#e7e7e7;
  --sans:'Inter Tight','Helvetica Neue',Arial,sans-serif; }
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background:#e6e6e6 !important; font-family:var(--sans); color:var(--ink); }
header[data-testid="stHeader"] { background:transparent; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
.block-container { max-width:none !important; width:calc(100% - 1.4rem) !important; background:#fff; border-radius:30px;
  margin:.7rem auto !important; padding:1.5rem clamp(1.1rem,3vw,3.4rem) 2.8rem !important; }
em { font-style:italic; font-weight:500; }
a { color:inherit; }
[data-testid="stMarkdownContainer"] a { text-decoration:none !important; color:inherit !important; }
[data-testid="stMarkdownContainer"] a.btn { color:#fff !important; }

/* nav */
.nav { display:flex; gap:.55rem; align-items:stretch; }
.nb { background:var(--soft); border-radius:14px; height:46px; display:flex; align-items:center; justify-content:center;
  padding:0 1.1rem; font-size:.66rem; text-transform:uppercase; letter-spacing:.05em; color:var(--ink);
  text-decoration:none; white-space:nowrap; transition:background .15s; }
a.nb:hover { background:var(--soft2); }
.nb.grow { flex:1; justify-content:space-between; }
.nb.logo b { font-size:.95rem; text-transform:none; letter-spacing:-.02em; font-weight:600; }
.nb.logo span { font-size:.46rem; line-height:1.25; text-align:right; color:var(--mid); }
.nb.dots { letter-spacing:.22em; font-size:1.05rem; padding:0 1.6rem; }
@media (max-width:860px){ .nb.hide{display:none;} }

/* hero */
.hero { position:relative; padding:3.4rem 0 2.4rem; }
.eyebrow { font-size:.7rem; text-transform:uppercase; letter-spacing:.04em; color:#555; margin-left:9%; position:relative; z-index:2; }
.h1 .l1 { display:inline-block; padding-right:clamp(3.2rem,4.6vw,6rem); }
.h1 { text-align:right; font-size:clamp(2.3rem,6.4vw,7.2rem); line-height:1.04; letter-spacing:-.04em; font-weight:500; margin:.2rem 0 0; position:relative; z-index:1; }
.av { position:absolute; border-radius:50%; object-fit:cover; border:3px solid #fff; box-shadow:0 6px 18px rgba(0,0,0,.2); z-index:3; background:#ddd; }
.a1 { width:clamp(56px,4.4vw,92px); height:clamp(56px,4.4vw,92px); left:3%; top:4.4rem; }
.a2 { width:clamp(50px,3.9vw,84px); height:clamp(50px,3.9vw,84px); right:1%; top:4.7rem; }
.h1 .l2 { position:relative; display:inline-block; }
.a3 { width:clamp(48px,3.7vw,80px); height:clamp(48px,3.7vw,80px); left:-.6rem; top:88%; }
.sub { margin:1.7rem 0 1.4rem 38%; max-width:36rem; color:var(--mid); font-size:clamp(1rem,1.15vw,1.3rem); line-height:1.45; letter-spacing:-.01em; }
.sub .dk { color:var(--ink); }
.cta { margin-left:38%; display:flex; align-items:center; gap:.5rem; }
.btn { display:inline-block; background:var(--ink); color:#fff !important; border-radius:10px; padding:.85rem 1.35rem; font-size:.7rem;
  text-transform:uppercase; letter-spacing:.05em; text-decoration:none; }
.circ { display:inline-flex; width:38px; height:38px; border:1px solid #cfcfcf; border-radius:50%; align-items:center; justify-content:center;
  text-decoration:none; font-size:.9rem; }
@media (max-width:820px){ .sub, .cta{margin-left:0;} .av{display:none;} .h1{text-align:left;} .eyebrow{margin-left:0;} }

/* panels */
.st-key-gen { background:var(--soft); border-radius:34px; padding:1.8rem 1.4rem 1.6rem; margin-top:1rem; }
.st-key-blend { border:1px solid var(--line); border-radius:34px; padding:1.8rem 1.4rem 1.6rem; margin-top:1.2rem; }
.ph { display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; flex-wrap:wrap; margin-bottom:1.1rem; }
.ph .t { font-size:clamp(1.7rem,3.3vw,2.5rem); letter-spacing:-.04em; font-weight:500; line-height:1.05; }
.chips { display:flex; gap:.4rem; flex-wrap:wrap; }
.chip { background:#fff; border:1px solid var(--line); border-radius:8px; padding:.32rem .65rem; font-size:.6rem;
  text-transform:uppercase; letter-spacing:.05em; color:#444; }
.st-key-blend .chip { background:var(--soft); border-color:transparent; }

/* widgets */
[data-testid="stWidgetLabel"] p { font-size:.64rem !important; text-transform:uppercase; letter-spacing:.06em; color:var(--mid) !important; font-family:var(--sans); }
.stButton > button, .stDownloadButton > button { border-radius:10px !important; font-family:var(--sans); font-weight:500; font-size:.7rem;
  text-transform:uppercase; letter-spacing:.05em; padding:.8rem 1.4rem; transition:all .15s; }
.stButton > button { background:var(--ink) !important; color:#fff !important; border:1px solid var(--ink) !important; }
.stButton > button p, .stDownloadButton > button p { color:inherit !important; font-weight:500; font-size:.7rem; }
.stButton > button:hover { background:#fff !important; color:var(--ink) !important; }
.stDownloadButton > button { background:#fff !important; color:var(--ink) !important; border:1px solid #cfcfcf !important; }
.stDownloadButton > button:hover { background:var(--ink) !important; color:#fff !important; border-color:var(--ink) !important; }
[data-baseweb="input"] { border-radius:12px !important; background:#fff !important; border:1px solid var(--line) !important; }
[data-baseweb="base-input"] { background:transparent !important; }
.stNumberInput button { background:transparent !important; color:var(--ink) !important; border-radius:0 !important; }
[data-testid="stRadio"] [role="radiogroup"] label p { font-size:.78rem; text-transform:none; letter-spacing:0; color:var(--ink) !important; }

/* face cards */
.fgrid { display:grid; grid-template-columns:repeat(6,1fr); gap:.8rem; margin-top:1.1rem; }
@media (min-width:1750px){ .fgrid{grid-template-columns:repeat(12,1fr);} .fc .chip{display:none;} .fc .lab{font-size:.8rem;}
  .fc .num{width:22px; height:22px; font-size:.55rem;} }
@media (max-width:1000px){ .fgrid{grid-template-columns:repeat(4,1fr);} .fc .chip{display:none;} }
@media (max-width:640px){ .fgrid{grid-template-columns:repeat(3,1fr);} }
.fc { position:relative; aspect-ratio:1/1.12; border-radius:20px; overflow:hidden; background:#ddd; }
.fc img { width:100%; height:100%; object-fit:cover; display:block; }
.fc .chip { position:absolute; left:.55rem; top:.55rem; background:rgba(255,255,255,.92); border:none; font-size:.5rem; padding:.25rem .5rem; }
.fc .num { position:absolute; right:.55rem; top:.55rem; width:26px; height:26px; border-radius:50%; background:#fff; display:flex;
  align-items:center; justify-content:center; font-size:.62rem; }
.fc .lab { position:absolute; left:0; bottom:0; background:#fff; border-radius:0 18px 0 0; padding:.4rem .8rem .3rem .65rem;
  font-size:.95rem; letter-spacing:-.03em; font-weight:500; line-height:1.1; }
.pline { height:2px; background:#d8d8d8; margin:1.6rem 8% 1.1rem; position:relative; }
.pline span { position:absolute; left:0; top:0; height:100%; width:22%; background:var(--ink); }

/* blend filmstrip */
.film { display:flex; gap:.4rem; margin-top:1.1rem; overflow-x:auto; }
.fr { position:relative; flex:1 0 64px; aspect-ratio:1/1.25; border-radius:16px; overflow:hidden; background:#ddd; }
.fr img { width:100%; height:100%; object-fit:cover; display:block; }
.fr .chip { position:absolute; left:.45rem; top:.45rem; background:rgba(255,255,255,.92); border:none; padding:.22rem .45rem; font-size:.55rem; }

/* how it works */
.svc { position:relative; display:grid; grid-template-columns:1fr 1.5fr 1fr; grid-template-rows:auto auto; gap:1.2rem 1rem;
  align-items:start; margin:4rem 0 2.5rem; min-height:clamp(540px,36vw,760px); }
.svc .big { grid-column:1/4; grid-row:1/3; align-self:center; justify-self:center; font-size:clamp(3.2rem,10.5vw,12rem);
  font-weight:500; letter-spacing:-.05em; line-height:1; white-space:nowrap; z-index:0; }
.sc { position:relative; z-index:1; background:var(--soft); border-radius:26px; padding:1.4rem 1.2rem 1.2rem; text-align:center; width:min(clamp(240px,19vw,340px),100%); }
.sc.w { background:#fff; border:1px solid var(--line); }
.sc .sav { width:42px; height:42px; border-radius:50%; object-fit:cover; margin-bottom:.8rem; }
.sc .st { font-size:1rem; line-height:1.15; letter-spacing:-.02em; margin-bottom:.55rem; }
.sc .sd { color:var(--mid); font-size:.68rem; line-height:1.5; margin-bottom:.9rem; }
.sc .rm { display:inline-block; background:#fff; border:1px solid #d5d5d5; border-radius:8px; padding:.4rem .9rem; font-size:.6rem; text-transform:uppercase; letter-spacing:.05em; }
.sc.w .rm { background:var(--soft); border-color:transparent; }
.c1 { grid-column:1; grid-row:1; justify-self:start; }
.c2 { grid-column:3; grid-row:1; justify-self:end; margin-top:-.5rem; }
.c3 { grid-column:1; grid-row:2; justify-self:start; }
.c4 { grid-column:3; grid-row:2; justify-self:end; }
@media (max-width:820px){ .svc{grid-template-columns:1fr; min-height:0;} .svc .big{grid-column:1; grid-row:auto; font-size:3.4rem;}
  .c1,.c2,.c3,.c4{grid-column:1; grid-row:auto; justify-self:stretch; width:auto; margin-top:0;} }

/* training progress */
.ctr { text-align:center; margin:4rem 0 1.6rem; }
.ctr .t { font-size:clamp(2.2rem,5.2vw,3.8rem); letter-spacing:-.045em; font-weight:500; line-height:1.05; }
.ctr .d { color:var(--mid); font-size:.95rem; margin-top:.8rem; }
.prog { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; }
@media (max-width:820px){ .prog{grid-template-columns:1fr;} }
.pc { background:var(--soft); border-radius:28px; padding:.9rem .9rem 1.1rem; }
.pc img { width:100%; aspect-ratio:1/1; object-fit:cover; border-radius:20px; display:block; }
.pc .cap { display:flex; justify-content:space-between; align-items:center; margin:.9rem .3rem 0; font-size:1.15rem; letter-spacing:-.03em; }
.loss { margin-top:1rem; background:var(--soft); border-radius:28px; padding:1.1rem; }
.loss img { width:100%; max-width:1200px; margin:0 auto; border-radius:18px; display:block; background:#fff; }
.loss .cap { color:var(--mid); font-size:.8rem; margin:.9rem .4rem .2rem; line-height:1.5; }

.footline { display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap; font-size:.66rem; text-transform:uppercase;
  letter-spacing:.05em; color:var(--mid); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

if not WEIGHTS.exists():
    st.error(f"Missing `{WEIGHTS}`. Train the model first and copy generator.pth there.")
    st.stop()


@st.cache_resource(show_spinner="Loading generator...")
def get_generator():
    return load_generator(WEIGHTS)


# ---------- helpers ----------
def face_img(arr_chw, size=256, gray=False):
    a = np.clip(np.asarray(arr_chw, dtype=np.float32), -1, 1)
    a = ((a + 1) / 2 * 255).astype(np.uint8).transpose(1, 2, 0)
    im = Image.fromarray(a).resize((size, size), Image.BICUBIC)
    return ImageOps.grayscale(im).convert("RGB") if gray else im


def uri(im, quality=88):
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def asset_uri(name, width=560):
    p = ASSETS / name
    if not p.exists():
        return None
    im = Image.open(p).convert("RGB")
    im.thumbnail((width, width * 3))
    return uri(im, 90)


def read_losses():
    out = {}
    try:
        with open(ASSETS / "losses.csv", newline="") as f:
            for row in csv.DictReader(f):
                out[int(row["epoch"])] = float(row["loss_G"])
    except Exception:
        pass
    return out


def png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


g = get_generator()
n_params = sum(p.numel() for p in g.parameters())

if "seed" not in st.session_state:
    st.session_state.seed = 42


def randomize():
    st.session_state.seed = random.randint(0, 1_000_000)


# ---------- nav + hero ----------
hero = generate(g, noise_from_seed(int(st.session_state.seed), 3))
av = [uri(face_img(hero[i], 160), 90) for i in range(3)]

st.markdown(
    f"""
<div class="nav">
  <div class="nb grow logo"><b>FaceGAN</b><span>DCGAN<br>PYTORCH</span></div>
  <a class="nb" href="#generate" target="_self">Generate</a>
  <a class="nb" href="#blend" target="_self">Blend</a>
  <a class="nb hide" href="#how" target="_self">How it works</a>
  <div class="nb dots hide">&bull;&bull;&bull;</div>
  <a class="nb hide" href="#progress" target="_self">Results</a>
</div>
<div class="hero">
  <div class="eyebrow">50,000 training faces</div>
  <div class="h1"><span class="l1">Dreaming up <em>faces</em> &amp;</span><br><span class="l2">creating <em>people</em> with FaceGAN<img class="av a3" src="{av[2]}" alt=""></span></div>
  <img class="av a1" src="{av[0]}" alt=""><img class="av a2" src="{av[1]}" alt="">
  <div class="sub"><span class="dk">How it works.</span>&nbsp; A generator turns random noise into a 64&times;64 face while a
    discriminator learns to spot fakes. Every person here is imaginary.</div>
  <div class="cta"><a class="btn" href="#generate" target="_self">Generate faces</a>
    <a class="circ" href="#generate" target="_self">&#8595;</a></div>
</div>
<div id="generate"></div>
""",
    unsafe_allow_html=True,
)

# ---------- generate panel ----------
with st.container(key="gen"):
    st.markdown(
        '<div class="ph"><div class="t">Latest <em>faces</em></div></div>', unsafe_allow_html=True
    )
    c1, c2, c3, c4 = st.columns([1.1, 1.4, 2, 1.6])
    c1.markdown("<div style='height:1.65rem'></div>", unsafe_allow_html=True)
    c1.button("Randomize", on_click=randomize, use_container_width=True)
    seed = c2.number_input("Seed (same seed = same faces)", 0, 10**9, step=1, key="seed")
    n = c3.slider("Number of faces", 12, 48, 12, step=12)
    bw = c4.radio("Display", ["Color", "Black & white"], horizontal=True) == "Black & white"

    faces = generate(g, noise_from_seed(seed, n))
    cards = "".join(
        f'<div class="fc"><img src="{uri(face_img(faces[i], 256, bw))}" alt="Generated face {i + 1}">'
        f'<span class="chip">AI generated</span><span class="num">{i + 1:02d}</span>'
        f'<div class="lab"><em>Face</em> {i + 1:02d}</div></div>'
        for i in range(n)
    )
    st.markdown(
        f"""<div class="chips" style="margin-top:1.2rem"><span class="chip">Seed {seed}</span>
<span class="chip">{n} faces</span><span class="chip">64&times;64 upscaled</span></div>
<div class="fgrid">{cards}</div><div class="pline"><span></span></div>""",
        unsafe_allow_html=True,
    )
    grid = to_pil_grid(faces, nrow=6)
    grid = ImageOps.grayscale(grid) if bw else grid
    st.download_button("Download PNG", png_bytes(grid), f"faces_seed{seed}.png", "image/png")

# ---------- blend panel ----------
st.markdown('<div id="blend"></div>', unsafe_allow_html=True)
with st.container(key="blend"):
    st.markdown(
        '<div class="ph"><div class="t">Blend two faces <em>smoothly</em></div></div>',
        unsafe_allow_html=True,
    )
    b1, b2, b3 = st.columns([1, 1, 2])
    a = b1.number_input("Seed A", 0, 10**9, 10, step=1)
    b = b2.number_input("Seed B", 0, 10**9, 500, step=1)
    steps = b3.slider("Steps", 5, 12, 10)
    z = slerp(noise_from_seed(a)[0], noise_from_seed(b)[0], steps)
    frames = generate(g, z)
    film = "".join(
        f'<div class="fr" title="t = {i / (steps - 1):.2f}"><img src="{uri(face_img(frames[i], 200, bw))}" alt="">'
        + ('<span class="chip">A</span>' if i == 0 else '<span class="chip">B</span>' if i == steps - 1 else "")
        + "</div>"
        for i in range(steps)
    )
    st.markdown(
        f"""<div class="chips" style="margin-top:1.1rem"><span class="chip">Seed A {a}</span>
<span class="chip">Seed B {b}</span><span class="chip">{steps} steps</span>
<span class="chip">Spherical interpolation</span></div>
<div class="film">{film}</div><div class="pline"><span></span></div>""",
        unsafe_allow_html=True,
    )
    row = to_pil_grid(frames, nrow=steps, scale=1)
    row = ImageOps.grayscale(row) if bw else row
    st.download_button("Download PNG", png_bytes(row), f"blend_{a}_{b}.png", "image/png")

# ---------- how it works ----------
svc = generate(g, noise_from_seed(2024, 4))
sv = [uri(face_img(svc[i], 120), 90) for i in range(4)]


def card(cls, img, title, desc, tag):
    return (
        f'<div class="sc {cls}"><img class="sav" src="{img}" alt=""><div class="st">{title}</div>'
        f'<div class="sd">{desc}</div><span class="rm">{tag}</span></div>'
    )


st.markdown(
    f"""
<div id="how"></div>
<div class="svc">
  <div class="big">How it <em>works</em></div>
  {card("c1", sv[0], "Generator", "Turns 100 random numbers into a 64&times;64 face using five transposed convolutions.", f"{n_params / 1e6:.1f}M parameters")}
  {card("c2 w", sv[1], "Discriminator", "A second network that guesses whether a face is real or fake, pushing the generator to improve.", "Real vs fake")}
  {card("c3 w", sv[2], "Training", "30 epochs on 50,000 CelebA photos. Label smoothing helped keep both networks in balance.", "30 epochs")}
  {card("c4", sv[3], "Latent space", "Every seed is a point in noise space. Nearby points give similar faces, which is what Blend shows.", "100 dimensions")}
</div>
""",
    unsafe_allow_html=True,
)

# ---------- training progress (uses the images saved in ./assets) ----------
snaps = [(1, "epoch_001.png"), (15, "epoch_015.png"), (30, "epoch_030.png")]
loaded = [(e, asset_uri(f)) for e, f in snaps]
if all(u for _, u in loaded):
    gl = read_losses()
    pcs = "".join(
        f'<div class="pc"><img src="{u}" alt="Samples after epoch {e}"><div class="cap"><span><em>Epoch</em> {e:02d}</span>'
        + (f'<span class="chip">G loss {gl[e]:.2f}</span>' if e in gl else "")
        + "</div></div>"
        for e, u in loaded
    )
    loss_uri = asset_uri("loss_curves.png", 1350)
    loss_card = (
        f'<div class="loss"><img src="{loss_uri}" alt="Training loss curves"><div class="cap">Generator and discriminator '
        f"losses over 11,700 training iterations. Both settle rather than one overpowering the other.</div></div>"
        if loss_uri
        else ""
    )
    st.markdown(
        f"""
<div id="progress"></div>
<div class="ctr"><div class="t">Training <em>progress</em></div>
<div class="d">The same 64 noise vectors, shown after epochs 1, 15 and 30.</div></div>
<div class="prog">{pcs}</div>{loss_card}
""",
        unsafe_allow_html=True,
    )

st.markdown(
    """
<div class="pline" style="margin-top:2.6rem"><span></span></div>
<div class="footline"><span>Built with PyTorch &middot; Streamlit &middot; Trained on CelebA</span>
<a href="#generate" target="_self" style="text-decoration:none">Back to top &uarr;</a></div>
""",
    unsafe_allow_html=True,
)