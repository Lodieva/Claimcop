"""
ClaimCop - Car Damage Detection webapp
Run met: streamlit run app/main.py
"""

import base64
import sys
from io import BytesIO
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
from PIL import Image

from src.download_model import download_baseline_model
from src.inference import detect_damage

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_PATH = ASSETS_DIR / "claimcop_logo.png"
ICON_PATH = ASSETS_DIR / "claimcop_icon.png"

st.set_page_config(
    page_title="ClaimCop — Car Damage Detection",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else "🚗",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def img_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@st.cache_resource(show_spinner="Model wordt gedownload (eenmalig, kan even duren)...")
def ensure_model_available():
    """Zorgt dat het model lokaal staat, ook op een lege hosting-container."""
    return download_baseline_model()


def severity_badge(area_pct: float | None) -> str:
    """
    Geeft een kleurgecodeerd ernst-label terug op basis van het geschatte
    schade-oppervlak. Drempels zijn een eerste, redelijke inschatting —
    stem deze in de praktijk af met een schade-expert.
    """
    if area_pct is None:
        return ""
    if area_pct < 2:
        return '<span class="cc-severity cc-severity--low">Licht</span>'
    if area_pct < 8:
        return '<span class="cc-severity cc-severity--medium">Matig</span>'
    return '<span class="cc-severity cc-severity--high">Ernstig</span>'


# ---------------------------------------------------------------------------
# Styling — ClaimCop design system
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

    :root {
        --ink: #1A2839;
        --ink-soft: #55677E;
        --accent: #1367D5;
        --accent-soft: #E8F0FD;
        --bg: #F5F7FA;
        --surface: #FFFFFF;
        --border: #E2E8F0;
    }

    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    .stApp { background: var(--bg); }

    /* Upload button fix done by NR */
    [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded' !important;
    }

    h1, h2, h3 { font-family: 'Sora', sans-serif; color: var(--ink); letter-spacing: -0.01em; }

    #MainMenu, footer, header { visibility: hidden; }

    /* ---------- Header ---------- */
    .cc-header {
        display: flex; align-items: center; gap: 16px;
        padding: 8px 0 20px 0; margin-bottom: 8px;
        border-bottom: 1px solid var(--border);
    }
    .cc-header img { width: 56px; height: 56px; }
    .cc-wordmark { font-family: 'Sora', sans-serif; font-weight: 800; font-size: 28px; color: var(--ink); line-height: 1.1; }
    .cc-wordmark span { color: var(--accent); }
    .cc-tagline {
        font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.08em;
        text-transform: uppercase; color: var(--ink-soft); margin-top: 2px;
    }

    /* ---------- Scan-frame kader (signature element uit het logo) ---------- */
    .cc-scanframe { position: relative; padding: 14px; }
    .cc-scanframe::before, .cc-scanframe::after,
    .cc-scanframe .corner-tr, .cc-scanframe .corner-bl {
        content: ""; position: absolute; width: 22px; height: 22px;
        border: 3px solid var(--accent);
    }
    .cc-scanframe::before { top: 0; left: 0; border-right: none; border-bottom: none; }
    .cc-scanframe::after { bottom: 0; right: 0; border-left: none; border-top: none; }
    .cc-scanframe .corner-tr { top: 0; right: 0; border-left: none; border-bottom: none; }
    .cc-scanframe .corner-bl { bottom: 0; left: 0; border-right: none; border-top: none; }
    .cc-scanframe img { width: 100%; border-radius: 4px; display: block; }

    /* ---------- Cards ---------- */
    .cc-card {
        background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
        padding: 20px; margin-bottom: 16px;
    }
    .cc-card-title { font-family: 'Sora', sans-serif; font-weight: 700; font-size: 16px; color: var(--ink); margin-bottom: 12px; }

    /* ---------- Detectie-rij ---------- */
    .cc-detection { display: flex; align-items: center; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid var(--border); }
    .cc-detection:last-child { border-bottom: none; }
    .cc-detection-label { display: flex; align-items: center; gap: 10px; }
    .cc-chip {
        background: var(--accent-soft); color: var(--accent); font-family: 'IBM Plex Mono', monospace;
        font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;
        padding: 4px 10px; border-radius: 20px;
    }
    .cc-detection-meta { font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: var(--ink-soft); display: flex; gap: 16px; }
    .cc-detection-meta b { color: var(--ink); }

    /* ---------- Advies-balk ---------- */
    .cc-advice {
        border-left: 3px solid var(--accent); background: var(--accent-soft);
        padding: 12px 16px; border-radius: 0 8px 8px 0; font-size: 14px; color: var(--ink);
        margin-top: 8px;
    }
    .cc-advice b { font-family: 'Sora', sans-serif; }

    .cc-empty { color: var(--ink-soft); font-size: 14px; padding: 8px 0; }

    /* ---------- Ernst-badges ---------- */
    .cc-severity { font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.04em; padding: 4px 10px; border-radius: 20px; }
    .cc-severity--low { background: #E7F7EE; color: #17803D; }
    .cc-severity--medium { background: #FFF4E0; color: #B4690E; }
    .cc-severity--high { background: #FDEAEA; color: #C22B2B; }

    /* Streamlit-widgets meekleuren */
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 10px;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        position: relative; z-index: 2;
    }
    .stSlider label p { font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important; text-transform: uppercase; letter-spacing: 0.04em; color: var(--ink-soft) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

if ICON_PATH.exists():
    icon_b64 = img_to_base64(Image.open(ICON_PATH))
    st.markdown(
        f"""
        <div class="cc-header">
            <img src="data:image/png;base64,{icon_b64}" />
            <div>
                <div class="cc-wordmark">Claim<span>Cop</span></div>
                <div class="cc-tagline">See damage. Speed claims.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.title("🚗 ClaimCop")
    st.caption("See damage. Speed claims.")

ensure_model_available()

st.markdown(
    """
    Upload één of meerdere foto's van de schade. ClaimCop detecteert en
    classificeert zichtbare schade automatisch, als eerste inschatting
    vóór beoordeling door een schade-expert.
    """
)

conf_threshold = st.slider(
    "Zekerheidsdrempel",
    min_value=0.05, max_value=0.9, value=0.25, step=0.05,
    help="Lager = meer (mogelijk onzekere) detecties. Hoger = alleen zeer zekere detecties.",
)

uploaded_files = st.file_uploader(
    "Upload foto('s) van de schade",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

# ---------------------------------------------------------------------------
# Resultaten
# ---------------------------------------------------------------------------

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.divider()

        tmp_path = Path("data") / "processed" / uploaded_file.name
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.open(uploaded_file).convert("RGB")
        image.save(tmp_path)

        with st.spinner(f"Analyseren van {uploaded_file.name}..."):
            results, summary = detect_damage(tmp_path, conf=conf_threshold)

        annotated_np = results.plot()[:, :, ::-1]
        annotated_img = Image.fromarray(annotated_np)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div class="cc-scanframe"><div class="corner-tr"></div><div class="corner-bl"></div>'
                f'<img src="data:image/png;base64,{img_to_base64(image)}" /></div>',
                unsafe_allow_html=True,
            )
            st.caption("Origineel")
        with col2:
            st.markdown(
                f'<div class="cc-scanframe"><div class="corner-tr"></div><div class="corner-bl"></div>'
                f'<img src="data:image/png;base64,{img_to_base64(annotated_img)}" /></div>',
                unsafe_allow_html=True,
            )
            st.caption("Detectie")

        rows_html = ""
        for item in summary:
            area_html = f'<span>oppervlak: <b>{item["area_pct"]}%</b></span>' if item.get("area_pct") is not None else ""
            badge = severity_badge(item.get("area_pct"))
            rows_html += (
                '<div class="cc-detection">'
                '<div class="cc-detection-label">'
                f'<span class="cc-chip">{item["class"]}</span>'
                f'{badge}'
                '</div>'
                '<div class="cc-detection-meta">'
                f'<span>confidence: <b>{item["confidence"]:.0%}</b></span>'
                f'{area_html}'
                '</div>'
                '</div>'
            )

        if summary:
            card_html = (
                '<div class="cc-card">'
                f'<div class="cc-card-title">Resultaat — {uploaded_file.name}</div>'
                f'{rows_html}'
                '<div class="cc-advice"><b>Advies:</b> laat deze claim beoordelen door een schade-expert.</div>'
                '</div>'
            )
        else:
            card_html = (
                '<div class="cc-card">'
                f'<div class="cc-card-title">Resultaat — {uploaded_file.name}</div>'
                '<div class="cc-empty">Geen schade gedetecteerd bij deze zekerheidsdrempel.</div>'
                '</div>'
            )
        st.markdown(card_html, unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="cc-empty">⬆️ Upload een foto om te starten.</div>',
        unsafe_allow_html=True,
    )