import os
from datetime import datetime
from typing import Optional
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from huggingface_hub import InferenceClient
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="SpecSense AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# CONSTANTS
# =========================
APP_NAME = "SpecSense AI"
APP_VERSION = "V1.0"
G_SHEET_URL = "https://docs.google.com/spreadsheets/d/1Xy4tgkGs1OXOTh-OMAsR7YsfkUPxttF7qalhDdhHa90/export?format=csv&gid=0"
G_SCRIPT_URL ="https://script.google.com/macros/s/AKfycbzhJ9Tep6M55eI-XX1t-0jb9wglnOwL8nICcRX1U5XReXpJKCWBEnuMn9zgpx_aYjKd3A/exec"
LOGO_PATH = "logo.png"
PDF_PATH = "rapport_qualite_specsense.pdf"

MENU_ITEMS = [
    "➕ Saisie Mesures",
    "🏠 Tableau de bord",
    "📏 MSA",
    "📉 SPC",
    "🎯 Capabilité",
    "📊 Pareto",
    "⚠️ AMDEC",
    "🤖 IA",
]

REQUIRED_COLS = [
    "Date_Time",
    "Part_ID",
    "Operator",
    "Trial",
    "Measurement",
    "USL",
    "LSL",
    "Machine",
    "Defect_Type",
    "Severity",
    "Occurrence",
    "Detection",
]

NUMERIC_COLS = ["Measurement", "USL", "LSL", "Severity", "Occurrence", "Detection"]


# =========================
# CSS
# =========================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    /* ========== BASE ========== */
    html, body, .stApp {
        font-family: 'Inter', sans-serif !important;
        background: #020917 !important;
        color: #e2e8f0 !important;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, rgba(14,165,233,0.07) 0%, transparent 60%),
            radial-gradient(ellipse at 90% 100%, rgba(99,102,241,0.06) 0%, transparent 60%),
            linear-gradient(180deg, #020917 0%, #040f1e 100%) !important;
    }

    /* ========== BLOCK CONTAINER ========== */
    .block-container {
        padding-top: 1.5rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 100% !important;
    }

    /* ========== SIDEBAR ========== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020917 0%, #050e1d 60%, #040c1a 100%) !important;
        border-right: 1px solid rgba(56,189,248,0.12) !important;
        box-shadow: 4px 0 24px rgba(0,0,0,0.4) !important;
    }

    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #38bdf8 !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        margin-bottom: 8px !important;
    }

    section[data-testid="stSidebar"] .stCaption {
        color: #475569 !important;
        font-size: 11px !important;
    }

    /* ========== RADIO MENU ========== */
    div[role="radiogroup"] label {
        background: rgba(15,23,42,0.5) !important;
        border: 1px solid rgba(148,163,184,0.1) !important;
        border-radius: 12px !important;
        padding: 11px 14px !important;
        margin-bottom: 6px !important;
        transition: all 0.2s ease !important;
        color: #94a3b8 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }

    div[role="radiogroup"] label:hover {
        background: rgba(14,165,233,0.12) !important;
        border-color: rgba(56,189,248,0.4) !important;
        color: #38bdf8 !important;
        transform: translateX(4px) !important;
    }

    div[role="radiogroup"] label[data-checked="true"],
    div[role="radiogroup"] label:has(input:checked) {
        background: rgba(14,165,233,0.15) !important;
        border-color: rgba(56,189,248,0.5) !important;
        color: #38bdf8 !important;
        font-weight: 700 !important;
        border-left: 3px solid #38bdf8 !important;
    }

    /* ========== METRICS ========== */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(22,33,54,0.85)) !important;
        border: 1px solid rgba(56,189,248,0.2) !important;
        border-radius: 20px !important;
        padding: 20px 22px !important;
        box-shadow:
            0 8px 32px rgba(0,0,0,0.3),
            inset 0 1px 0 rgba(255,255,255,0.04) !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }

    div[data-testid="stMetric"]:hover {
        border-color: rgba(56,189,248,0.4) !important;
        box-shadow: 0 12px 40px rgba(0,0,0,0.4), 0 0 20px rgba(56,189,248,0.06) !important;
    }

    div[data-testid="stMetricLabel"] p {
        color: #64748b !important;
        font-weight: 600 !important;
        font-size: 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-size: 30px !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
    }

    div[data-testid="stMetricDelta"] {
        font-size: 12px !important;
        font-weight: 600 !important;
    }

    /* ========== TABS ========== */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15,23,42,0.6) !important;
        border-radius: 14px !important;
        padding: 4px !important;
        border: 1px solid rgba(56,189,248,0.12) !important;
        gap: 2px !important;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 10px !important;
        color: #64748b !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 8px 16px !important;
        transition: all 0.2s !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #38bdf8 !important;
        background: rgba(56,189,248,0.08) !important;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(56,189,248,0.15) !important;
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* ========== BUTTONS ========== */
    .stButton > button {
        background: linear-gradient(135deg, rgba(14,165,233,0.15), rgba(99,102,241,0.12)) !important;
        border: 1px solid rgba(56,189,248,0.35) !important;
        border-radius: 12px !important;
        color: #38bdf8 !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        padding: 10px 22px !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.02em !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(14,165,233,0.25), rgba(99,102,241,0.2)) !important;
        border-color: rgba(56,189,248,0.6) !important;
        box-shadow: 0 0 20px rgba(56,189,248,0.15) !important;
        transform: translateY(-1px) !important;
        color: #7dd3fc !important;
    }

    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* ========== FORM / INPUTS ========== */
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox select,
    .stTextArea textarea {
        background: rgba(15,23,42,0.8) !important;
        border: 1px solid rgba(56,189,248,0.2) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        font-size: 13px !important;
        transition: border-color 0.2s !important;
    }

    .stTextInput input:focus,
    .stNumberInput input:focus,
    .stTextArea textarea:focus {
        border-color: rgba(56,189,248,0.5) !important;
        box-shadow: 0 0 0 3px rgba(56,189,248,0.08) !important;
    }

    /* ========== SELECTBOX ========== */
    div[data-baseweb="select"] > div {
        background: rgba(15,23,42,0.8) !important;
        border: 1px solid rgba(56,189,248,0.2) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }

    /* ========== DATAFRAME ========== */
    .stDataFrame {
        border-radius: 14px !important;
        overflow: hidden !important;
        border: 1px solid rgba(56,189,248,0.15) !important;
    }

    .stDataFrame thead th {
        background: rgba(14,165,233,0.1) !important;
        color: #38bdf8 !important;
        font-weight: 700 !important;
        font-size: 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        border-bottom: 1px solid rgba(56,189,248,0.2) !important;
    }

    .stDataFrame tbody tr {
        border-bottom: 1px solid rgba(255,255,255,0.03) !important;
        transition: background 0.15s !important;
    }

    .stDataFrame tbody tr:hover {
        background: rgba(56,189,248,0.05) !important;
    }

    .stDataFrame tbody td {
        color: #cbd5e1 !important;
        font-size: 13px !important;
    }

    /* ========== ALERTS ========== */
    .stSuccess {
        background: rgba(34,197,94,0.1) !important;
        border: 1px solid rgba(34,197,94,0.3) !important;
        border-radius: 12px !important;
        color: #86efac !important;
    }

    .stWarning {
        background: rgba(245,158,11,0.1) !important;
        border: 1px solid rgba(245,158,11,0.3) !important;
        border-radius: 12px !important;
        color: #fcd34d !important;
    }

    .stError {
        background: rgba(239,68,68,0.1) !important;
        border: 1px solid rgba(239,68,68,0.3) !important;
        border-radius: 12px !important;
        color: #fca5a5 !important;
    }

    .stInfo {
        background: rgba(56,189,248,0.08) !important;
        border: 1px solid rgba(56,189,248,0.25) !important;
        border-radius: 12px !important;
        color: #7dd3fc !important;
    }

    /* ========== SUBHEADER / TITLES ========== */
    h1, h2, h3 {
        color: #f1f5f9 !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
    }

    .stMarkdown h3 {
        color: #e2e8f0 !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        padding-bottom: 8px !important;
        border-bottom: 1px solid rgba(56,189,248,0.15) !important;
        margin-bottom: 16px !important;
    }

    /* ========== DIVIDER ========== */
    hr {
        border: none !important;
        border-top: 1px solid rgba(56,189,248,0.12) !important;
        margin: 20px 0 !important;
    }

    /* ========== SPINNER ========== */
    .stSpinner > div {
        border-top-color: #38bdf8 !important;
    }

    /* ========== SCROLLBAR ========== */
    ::-webkit-scrollbar {
        width: 5px !important;
        height: 5px !important;
    }

    ::-webkit-scrollbar-track {
        background: transparent !important;
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(56,189,248,0.25) !important;
        border-radius: 3px !important;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(56,189,248,0.45) !important;
    }

    /* ========== PRO CARD ========== */
    .pro-card {
        background: linear-gradient(135deg, rgba(15,23,42,0.97), rgba(22,33,54,0.85)) !important;
        border: 1px solid rgba(56,189,248,0.2) !important;
        border-radius: 20px !important;
        padding: 22px 24px !important;
        box-shadow: 0 16px 40px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04) !important;
        margin-bottom: 16px !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }

    .pro-card:hover {
        border-color: rgba(56,189,248,0.35) !important;
        box-shadow: 0 20px 50px rgba(0,0,0,0.4), 0 0 30px rgba(56,189,248,0.05) !important;
    }

    /* ========== STATUS BADGES ========== */
    .status-ok {
        color: #4ade80 !important;
        font-weight: 800 !important;
        background: rgba(34,197,94,0.1) !important;
        padding: 3px 10px !important;
        border-radius: 20px !important;
        border: 1px solid rgba(34,197,94,0.3) !important;
        font-size: 12px !important;
    }

    .status-warning {
        color: #fbbf24 !important;
        font-weight: 800 !important;
        background: rgba(245,158,11,0.1) !important;
        padding: 3px 10px !important;
        border-radius: 20px !important;
        border: 1px solid rgba(245,158,11,0.3) !important;
        font-size: 12px !important;
    }

    .status-bad {
        color: #f87171 !important;
        font-weight: 800 !important;
        background: rgba(239,68,68,0.1) !important;
        padding: 3px 10px !important;
        border-radius: 20px !important;
        border: 1px solid rgba(239,68,68,0.3) !important;
        font-size: 12px !important;
    }

    /* ========== FORM SUBMIT BUTTON ========== */
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
        border: none !important;
        border-radius: 12px !important;
        color: white !important;
        font-weight: 800 !important;
        font-size: 14px !important;
        padding: 12px 28px !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 20px rgba(14,165,233,0.3) !important;
        letter-spacing: 0.02em !important;
    }

    .stFormSubmitButton > button:hover {
        box-shadow: 0 8px 30px rgba(14,165,233,0.45) !important;
        transform: translateY(-2px) !important;
        opacity: 0.95 !important;
    }

    /* ========== PLOTLY CHARTS ========== */
    .js-plotly-plot .plotly .modebar {
        background: rgba(15,23,42,0.8) !important;
        border-radius: 8px !important;
    }

    .js-plotly-plot .plotly .modebar-btn path {
        fill: #64748b !important;
    }

    .js-plotly-plot .plotly .modebar-btn:hover path {
        fill: #38bdf8 !important;
    }

    /* ========== CAPTION ========== */
    .stCaption {
        color: #475569 !important;
        font-size: 11px !important;
    }

    /* ========== DOWNLOAD BUTTON ========== */
    .stDownloadButton > button {
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(14,165,233,0.12)) !important;
        border: 1px solid rgba(99,102,241,0.35) !important;
        border-radius: 12px !important;
        color: #a5b4fc !important;
        font-weight: 700 !important;
        transition: all 0.2s !important;
    }

    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(14,165,233,0.2)) !important;
        border-color: rgba(99,102,241,0.6) !important;
        box-shadow: 0 0 20px rgba(99,102,241,0.2) !important;
        transform: translateY(-1px) !important;
    }

    </style>
    """, unsafe_allow_html=True)

# =========================
# HELPERS
# =========================
def plot_chart(fig: go.Figure, key: str, height: Optional[int] = None) -> None:
    if height is not None:
        fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True, key=key)


def safe_std(series: pd.Series) -> float:
    value = series.std()
    return 0.0 if pd.isna(value) else float(value)


def process_status(cpk: float) -> None:
    if cpk < 1:
        st.error("🚨 Statut global : Processus non capable")
    elif cpk < 1.33:
        st.warning("⚠️ Statut global : Amélioration nécessaire")
    else:
        st.success("✅ Statut global : Processus capable")


def clean_page_name(page: str) -> str:
    return page.split(" ", 1)[1] if " " in page else page


# =========================
# AI FUNCTIONS
# =========================
def ask_hf_ai(question: str) -> str:
    if "HUGGINGFACE_TOKEN" not in st.secrets:
        return "❌ HUGGINGFACE_TOKEN manquant dans Streamlit Secrets."

    try:
        client = InferenceClient(token=st.secrets["HUGGINGFACE_TOKEN"])
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un expert qualité automobile. Réponds en français simple, professionnel et avec des actions concrètes.",
                },
                {"role": "user", "content": question},
            ],
            max_tokens=650,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as exc:
        return f"❌ Erreur IA : {exc}"


def generate_ai_module_analysis(module_name: str, context: str) -> str:
    prompt = f"""
Tu es un expert qualité automobile.

Module analysé : {module_name}

Données :
{context}

Réponds exactement avec ce format :

INTERPRÉTATION :
- ...

ACTIONS RECOMMANDÉES :
- ...
- ...
- ...

Réponse en français claire, professionnelle et concrète.
"""
    return ask_hf_ai(prompt)


def show_ai_analysis(module_name: str, context: str) -> None:
    st.markdown("### 🤖 Interprétation IA & Actions recommandées")
    cache_key = f"ai_{module_name}_{abs(hash(context))}"

    if cache_key not in st.session_state:
        with st.spinner(f"🤖 Analyse IA {module_name}..."):
            st.session_state[cache_key] = generate_ai_module_analysis(module_name, context)

    st.info(st.session_state[cache_key])


# =========================
# DATA
# =========================
@st.cache_data(ttl=60)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(G_SHEET_URL)
    return validate_and_clean_data(df)


def validate_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()

    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Colonnes manquantes : {missing_cols}")
        return pd.DataFrame(columns=REQUIRED_COLS)

    for col in NUMERIC_COLS:
        df[col] = df[col].astype(str).str.replace(",", ".", regex=False).str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid_rows = df[df[NUMERIC_COLS].isna().any(axis=1)]
    if not invalid_rows.empty:
        st.error("❌ Erreur data : certaines valeurs numériques sont invalides.")
        st.dataframe(invalid_rows, use_container_width=True)
        st.stop()

    return df

def clean_for_json(value):
    if pd.isna(value):
        return ""

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%d %H:%M:%S")

    if hasattr(value, "item"):
        return value.item()

    return value


def save_to_google_sheet(row: dict) -> None:
    try:
        clean_row = {key: clean_for_json(value) for key, value in row.items()}

        response = requests.post(
            G_SCRIPT_URL,
            json=clean_row,
            timeout=10
        )

        if response.status_code == 200:
            st.success("✅ Sauvegardé dans Google Sheet")
        else:
            st.error(f"❌ Erreur Google Sheet: {response.text}")

    except Exception as e:
        st.error(f"❌ Erreur sauvegarde Google Sheet: {e}")


def prepare_data(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "msa_data": pd.DataFrame(columns=REQUIRED_COLS),
            "spc_data": pd.DataFrame(columns=REQUIRED_COLS),
            "total": 0,
            "msa_count": 0,
            "spc_count": 0,
            "mean_val": 0.0,
            "std_val": 0.0,
            "usl": 0.0,
            "lsl": 0.0,
            "cp": 0.0,
            "cpk": 0.0,
        }

    msa_data = df[df["Part_ID"].astype(str).str.contains("MSA", case=False, na=False)].copy()
    spc_data = df[df["Part_ID"].astype(str).str.contains("SPC", case=False, na=False)].copy()

    if spc_data.empty:
        spc_data = df.copy()

    mean_val = float(df["Measurement"].mean())
    std_val = safe_std(df["Measurement"])
    usl = float(df["USL"].iloc[0])
    lsl = float(df["LSL"].iloc[0])

    if std_val > 0:
        cp = (usl - lsl) / (6 * std_val)
        cpk = min(
            (usl - mean_val) / (3 * std_val),
            (mean_val - lsl) / (3 * std_val)
        )
    else:
        cp = 0.0
        cpk = 0.0

    return {
        "msa_data": msa_data,
        "spc_data": spc_data,
        "total": len(df),
        "msa_count": len(msa_data),
        "spc_count": len(spc_data),
        "mean_val": mean_val,
        "std_val": std_val,
        "usl": usl,
        "lsl": lsl,
        "cp": cp,
        "cpk": cpk,
    }


# =========================
# PDF
# =========================
def generate_pdf_report(metrics: dict, df: pd.DataFrame) -> str:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak, Image
    )
    from reportlab.pdfgen import canvas
    from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate
    import io

    # ─── COULEURS ───────────────────────────────────────────────
    BLEU_FONCE   = colors.HexColor("#020917")
    BLEU_MED     = colors.HexColor("#0f172a")
    BLEU_ACCENT  = colors.HexColor("#0ea5e9")
    BLEU_LIGHT   = colors.HexColor("#38bdf8")
    INDIGO       = colors.HexColor("#6366f1")
    GRIS_TEXTE   = colors.HexColor("#e2e8f0")
    GRIS_MED     = colors.HexColor("#94a3b8")
    GRIS_DIM     = colors.HexColor("#475569")
    VERT         = colors.HexColor("#22c55e")
    ORANGE       = colors.HexColor("#f59e0b")
    ROUGE        = colors.HexColor("#ef4444")
    BLANC        = colors.white
    NOIR         = colors.HexColor("#020917")

    PAGE_W, PAGE_H = A4

    # ─── HEADER / FOOTER sur chaque page ────────────────────────
    def header_footer(canvas_obj, doc):
        canvas_obj.saveState()

        # Header background
        canvas_obj.setFillColor(BLEU_MED)
        canvas_obj.rect(0, PAGE_H - 2.2*cm, PAGE_W, 2.2*cm, fill=1, stroke=0)

        # Ligne accent header
        canvas_obj.setFillColor(BLEU_ACCENT)
        canvas_obj.rect(0, PAGE_H - 2.2*cm, PAGE_W, 0.12*cm, fill=1, stroke=0)

        # Logo / Nom app
        canvas_obj.setFillColor(BLANC)
        canvas_obj.setFont("Helvetica-Bold", 13)
        canvas_obj.drawString(1.5*cm, PAGE_H - 1.5*cm, "SpecSense AI")
        canvas_obj.setFillColor(GRIS_MED)
        canvas_obj.setFont("Helvetica", 9)
        canvas_obj.drawString(1.5*cm, PAGE_H - 1.9*cm, "Plateforme intelligente de qualite industrielle")

        # Date header droite
        canvas_obj.setFillColor(GRIS_MED)
        canvas_obj.setFont("Helvetica", 9)
        date_str = datetime.now().strftime("%d/%m/%Y  %H:%M")
        canvas_obj.drawRightString(PAGE_W - 1.5*cm, PAGE_H - 1.5*cm, date_str)
        canvas_obj.setFillColor(GRIS_DIM)
        canvas_obj.setFont("Helvetica", 8)
        canvas_obj.drawRightString(PAGE_W - 1.5*cm, PAGE_H - 1.9*cm, "IATF 16949 | Qualite 4.0")

        # Footer background
        canvas_obj.setFillColor(BLEU_MED)
        canvas_obj.rect(0, 0, PAGE_W, 1.2*cm, fill=1, stroke=0)

        # Ligne accent footer
        canvas_obj.setFillColor(INDIGO)
        canvas_obj.rect(0, 1.2*cm, PAGE_W, 0.08*cm, fill=1, stroke=0)

        # Texte footer
        canvas_obj.setFillColor(GRIS_DIM)
        canvas_obj.setFont("Helvetica", 8)
        canvas_obj.drawString(1.5*cm, 0.4*cm, "CONFIDENTIEL — Document qualite interne")
        canvas_obj.drawRightString(PAGE_W - 1.5*cm, 0.4*cm, f"Page {doc.page}")

        canvas_obj.restoreState()

    # ─── STYLES ─────────────────────────────────────────────────
    def make_styles():
        s = {}

        s["titre_principal"] = ParagraphStyle(
            "titre_principal",
            fontName="Helvetica-Bold",
            fontSize=26,
            textColor=BLANC,
            alignment=TA_CENTER,
            spaceAfter=6,
            leading=30,
        )
        s["sous_titre"] = ParagraphStyle(
            "sous_titre",
            fontName="Helvetica",
            fontSize=13,
            textColor=GRIS_MED,
            alignment=TA_CENTER,
            spaceAfter=4,
        )
        s["section_title"] = ParagraphStyle(
            "section_title",
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=BLEU_LIGHT,
            spaceBefore=14,
            spaceAfter=8,
            leading=18,
        )
        s["body"] = ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=10,
            textColor=GRIS_TEXTE,
            spaceAfter=4,
            leading=15,
        )
        s["body_bold"] = ParagraphStyle(
            "body_bold",
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=BLANC,
            spaceAfter=4,
            leading=15,
        )
        s["small"] = ParagraphStyle(
            "small",
            fontName="Helvetica",
            fontSize=8,
            textColor=GRIS_DIM,
            spaceAfter=2,
        )
        s["conclusion"] = ParagraphStyle(
            "conclusion",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=BLANC,
            spaceAfter=6,
            leading=16,
        )
        s["label_centre"] = ParagraphStyle(
            "label_centre",
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=GRIS_MED,
            alignment=TA_CENTER,
        )
        s["valeur_centre"] = ParagraphStyle(
            "valeur_centre",
            fontName="Helvetica-Bold",
            fontSize=18,
            textColor=BLANC,
            alignment=TA_CENTER,
            leading=22,
        )
        return s

    ST = make_styles()

    # ─── HELPERS ────────────────────────────────────────────────
    def hr(color=BLEU_ACCENT, thickness=0.5):
        return HRFlowable(
            width="100%", thickness=thickness,
            color=color, spaceAfter=8, spaceBefore=4
        )

    def section_header(titre):
        return [
            Paragraph(titre, ST["section_title"]),
            hr(),
        ]

    def kpi_box(label, valeur, couleur=BLEU_LIGHT):
        data = [
            [Paragraph(label, ST["label_centre"])],
            [Paragraph(str(valeur), ST["valeur_centre"])],
        ]
        t = Table(data, colWidths=[3.8*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), BLEU_MED),
            ("ROUNDEDCORNERS",(0,0), (-1,-1), [8]),
            ("BOX",           (0,0), (-1,-1), 0.8, couleur),
            ("TOPPADDING",    (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ("LEFTPADDING",   (0,0), (-1,-1), 6),
            ("RIGHTPADDING",  (0,0), (-1,-1), 6),
        ]))
        return t

    def statut_color(cpk_val):
        if cpk_val >= 1.33:
            return VERT, "CAPABLE"
        elif cpk_val >= 1.0:
            return ORANGE, "LIMITE"
        else:
            return ROUGE, "NON CAPABLE"

    # ─── DOCUMENT SETUP ─────────────────────────────────────────
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=1.5*cm,
        rightMargin=1.5*cm,
        topMargin=2.8*cm,
        bottomMargin=1.8*cm,
    )

    story = []
    cpk   = metrics["cpk"]
    cp    = metrics["cp"]
    mean_val = metrics["mean_val"]
    std_val  = metrics["std_val"]
    usl   = metrics["usl"]
    lsl   = metrics["lsl"]
    total = metrics["total"]
    couleur_statut, texte_statut = statut_color(cpk)

    # ════════════════════════════════════════════════════════════
    # PAGE 1 — PAGE DE GARDE
    # ════════════════════════════════════════════════════════════

    story.append(Spacer(1, 1.5*cm))

    # Bloc titre principal
    title_data = [[
        Paragraph("RAPPORT QUALITE", ST["titre_principal"]),
    ]]
    title_table = Table(title_data, colWidths=[17*cm])
    title_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), BLEU_MED),
        ("BOX",           (0,0), (-1,-1), 1.5, BLEU_ACCENT),
        ("TOPPADDING",    (0,0), (-1,-1), 18),
        ("BOTTOMPADDING", (0,0), (-1,-1), 18),
        ("LEFTPADDING",   (0,0), (-1,-1), 20),
        ("RIGHTPADDING",  (0,0), (-1,-1), 20),
    ]))
    story.append(title_table)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Analyse de capabilite processus", ST["sous_titre"]))
    story.append(Spacer(1, 0.8*cm))
    story.append(hr(INDIGO, 1.5))
    story.append(Spacer(1, 0.5*cm))

    # Infos rapport
    info_data = [
        ["Date de generation :", datetime.now().strftime("%d %B %Y — %H:%M")],
        ["Reference document :", f"QR-{datetime.now().strftime('%Y%m%d-%H%M')}"],
        ["Nombre de mesures :", str(total)],
        ["Limites :", f"USL = {usl:.4f}   |   LSL = {lsl:.4f}"],
        ["Tolerance :", f"{(usl - lsl):.4f}"],
    ]
    info_table = Table(info_data, colWidths=[5*cm, 12*cm])
    info_table.setStyle(TableStyle([
        ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",      (1,0), (1,-1), "Helvetica"),
        ("FONTSIZE",      (0,0), (-1,-1), 10),
        ("TEXTCOLOR",     (0,0), (0,-1), GRIS_MED),
        ("TEXTCOLOR",     (1,0), (1,-1), BLANC),
        ("ROWBACKGROUNDS",(0,0), (-1,-1), [BLEU_MED, colors.HexColor("#0d1a2e")]),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
        ("BOX",           (0,0), (-1,-1), 0.5, GRIS_DIM),
        ("LINEBELOW",     (0,0), (-1,-2), 0.3, colors.HexColor("#1e293b")),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.8*cm))

    # Statut global — badge grand
    statut_data = [[
        Paragraph("STATUT GLOBAL DU PROCESSUS", ST["label_centre"]),
        Paragraph(f"● {texte_statut}", ParagraphStyle(
            "statut_badge",
            fontName="Helvetica-Bold",
            fontSize=20,
            textColor=couleur_statut,
            alignment=TA_CENTER,
        )),
    ]]
    statut_table = Table(statut_data, colWidths=[6*cm, 11*cm])
    statut_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), BLEU_MED),
        ("BOX",           (0,0), (-1,-1), 2, couleur_statut),
        ("TOPPADDING",    (0,0), (-1,-1), 14),
        ("BOTTOMPADDING", (0,0), (-1,-1), 14),
        ("LEFTPADDING",   (0,0), (-1,-1), 16),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(statut_table)
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    # PAGE 2 — RESUME EXECUTIF + CAPABILITE
    # ════════════════════════════════════════════════════════════

    story += section_header("1. RESUME EXECUTIF")

    # KPIs en ligne
    kpis_row = [
        kpi_box("Cp",         f"{cp:.2f}",       BLEU_ACCENT if cp >= 1.33 else ORANGE if cp >= 1 else ROUGE),
        kpi_box("Cpk",        f"{cpk:.2f}",       VERT if cpk >= 1.33 else ORANGE if cpk >= 1 else ROUGE),
        kpi_box("Moyenne",    f"{mean_val:.4f}",  BLEU_LIGHT),
        kpi_box("Ecart-type", f"{std_val:.5f}",   INDIGO),
    ]
    kpi_table = Table([kpis_row], colWidths=[4.0*cm]*4, hAlign="LEFT")
    kpi_table.setStyle(TableStyle([
        ("LEFTPADDING",  (0,0), (-1,-1), 3),
        ("RIGHTPADDING", (0,0), (-1,-1), 3),
        ("VALIGN",       (0,0), (-1,-1), "TOP"),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 0.5*cm))

    # Interpretation automatique
    story += section_header("2. CAPABILITE PROCESSUS")

    cap_data = [
        ["Parametre",         "Valeur",              "Seuil requis",  "Statut"],
        ["Cp",                f"{cp:.4f}",            ">= 1.33",       "OK" if cp >= 1.33 else "NOK"],
        ["Cpk",               f"{cpk:.4f}",           ">= 1.33",       "OK" if cpk >= 1.33 else "NOK"],
        ["Moyenne",           f"{mean_val:.4f}",      f"Cible: {(usl+lsl)/2:.4f}", "—"],
        ["Ecart-type",        f"{std_val:.6f}",       "Minimiser",     "—"],
        ["USL",               f"{usl:.4f}",           "—",             "—"],
        ["LSL",               f"{lsl:.4f}",           "—",             "—"],
        ["Tolerance",         f"{(usl-lsl):.4f}",     "—",             "—"],
        ["Decalage / Cible",  f"{(mean_val-(usl+lsl)/2):.6f}", "~0",  "OK" if abs(mean_val-(usl+lsl)/2) < 0.01 else "Attention"],
        ["Nb mesures",        str(total),             ">= 30",         "OK" if total >= 30 else "Insuffisant"],
    ]

    def statut_cell_color(val):
        if val == "OK":   return VERT
        if val == "NOK":  return ROUGE
        if val == "Attention": return ORANGE
        return GRIS_MED

    cap_table = Table(cap_data, colWidths=[4.5*cm, 4*cm, 4.5*cm, 4*cm])
    cap_style = [
        # Header
        ("BACKGROUND",    (0,0), (-1,0), BLEU_ACCENT),
        ("TEXTCOLOR",     (0,0), (-1,0), BLANC),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,0), 10),
        ("ALIGN",         (0,0), (-1,0), "CENTER"),
        # Body
        ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",      (0,1), (-1,-1), 9),
        ("TEXTCOLOR",     (0,1), (0,-1), GRIS_MED),
        ("TEXTCOLOR",     (1,1), (2,-1), BLANC),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [BLEU_MED, colors.HexColor("#0d1a2e")]),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 10),
        ("BOX",           (0,0), (-1,-1), 0.5, GRIS_DIM),
        ("LINEBELOW",     (0,0), (-1,-2), 0.3, colors.HexColor("#1e293b")),
        ("ALIGN",         (1,1), (-1,-1), "CENTER"),
    ]
    # Colorier colonne statut
    for i, row in enumerate(cap_data[1:], 1):
        c = statut_cell_color(row[3])
        cap_style.append(("TEXTCOLOR", (3,i), (3,i), c))
        cap_style.append(("FONTNAME",  (3,i), (3,i), "Helvetica-Bold"))

    cap_table.setStyle(TableStyle(cap_style))
    story.append(cap_table)
    story.append(Spacer(1, 0.5*cm))

    # ════════════════════════════════════════════════════════════
    # PAGE 3 — MSA + SPC
    # ════════════════════════════════════════════════════════════
    story.append(PageBreak())

    # MSA
    msa_data_df = metrics.get("msa_data", pd.DataFrame())
    story += section_header("3. SYSTEME DE MESURE (MSA)")

    if not msa_data_df.empty:
        mean_msa = float(msa_data_df["Measurement"].mean())
        std_msa  = float(msa_data_df["Measurement"].std()) if len(msa_data_df) > 1 else 0
        ref      = (usl + lsl) / 2
        tolerance = usl - lsl
        cg  = (0.2 * tolerance) / (6 * std_msa) if std_msa > 0 else 0
        cgk = (0.1 * tolerance - abs(mean_msa - ref)) / (3 * std_msa) if std_msa > 0 else 0

        msa_table_data = [
            ["Indicateur", "Valeur", "Seuil",   "Statut"],
            ["Cg",         f"{cg:.2f}",  ">= 1.33", "OK" if cg  >= 1.33 else "NOK"],
            ["Cgk",        f"{cgk:.2f}", ">= 1.33", "OK" if cgk >= 1.33 else "NOK"],
            ["Moyenne MSA",f"{mean_msa:.4f}", f"Ref: {ref:.4f}", "—"],
            ["Ecart-type", f"{std_msa:.6f}", "Minimiser", "—"],
            ["Nb mesures MSA", str(len(msa_data_df)), ">= 25", "OK" if len(msa_data_df) >= 25 else "Insuffisant"],
        ]
    else:
        msa_table_data = [
            ["Indicateur", "Valeur", "Seuil", "Statut"],
            ["Donnees MSA", "Non disponibles", "—", "—"],
        ]

    msa_t = Table(msa_table_data, colWidths=[4.5*cm, 4*cm, 4.5*cm, 4*cm])
    msa_style = [
        ("BACKGROUND",    (0,0), (-1,0), INDIGO),
        ("TEXTCOLOR",     (0,0), (-1,0), BLANC),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,0), 10),
        ("ALIGN",         (0,0), (-1,0), "CENTER"),
        ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",      (0,1), (-1,-1), 9),
        ("TEXTCOLOR",     (0,1), (0,-1), GRIS_MED),
        ("TEXTCOLOR",     (1,1), (2,-1), BLANC),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [BLEU_MED, colors.HexColor("#0d1a2e")]),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 10),
        ("BOX",           (0,0), (-1,-1), 0.5, GRIS_DIM),
        ("ALIGN",         (1,1), (-1,-1), "CENTER"),
    ]
    if not msa_data_df.empty:
        for i, row in enumerate(msa_table_data[1:], 1):
            c = statut_cell_color(row[3])
            msa_style.append(("TEXTCOLOR", (3,i), (3,i), c))
            msa_style.append(("FONTNAME",  (3,i), (3,i), "Helvetica-Bold"))
    msa_t.setStyle(TableStyle(msa_style))
    story.append(msa_t)
    story.append(Spacer(1, 0.5*cm))

    # SPC
    story += section_header("4. CONTROLE STATISTIQUE DU PROCESSUS (SPC)")

    spc_data_df = metrics.get("spc_data", df)
    mean_spc = float(spc_data_df["Measurement"].mean())
    std_spc  = float(spc_data_df["Measurement"].std()) if len(spc_data_df) > 1 else 0
    ucl_spc  = mean_spc + 3 * std_spc
    lcl_spc  = mean_spc - 3 * std_spc
    out_ctrl = len(spc_data_df[(spc_data_df["Measurement"] > ucl_spc) | (spc_data_df["Measurement"] < lcl_spc)])

    spc_table_data = [
        ["Parametre",           "Valeur",              "Statut"],
        ["Ligne centrale (CL)", f"{mean_spc:.4f}",     "—"],
        ["UCL (+3sigma)",       f"{ucl_spc:.4f}",      "—"],
        ["LCL (-3sigma)",       f"{lcl_spc:.4f}",      "—"],
        ["Points hors controle",str(out_ctrl),         "OK" if out_ctrl == 0 else "NOK"],
        ["Nombre points SPC",   str(len(spc_data_df)), "—"],
    ]
    spc_t = Table(spc_table_data, colWidths=[6*cm, 6*cm, 5*cm])
    spc_style = [
        ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#0369a1")),
        ("TEXTCOLOR",     (0,0), (-1,0), BLANC),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,0), 10),
        ("ALIGN",         (0,0), (-1,0), "CENTER"),
        ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",      (0,1), (-1,-1), 9),
        ("TEXTCOLOR",     (0,1), (0,-1), GRIS_MED),
        ("TEXTCOLOR",     (1,1), (1,-1), BLANC),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [BLEU_MED, colors.HexColor("#0d1a2e")]),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 10),
        ("BOX",           (0,0), (-1,-1), 0.5, GRIS_DIM),
        ("ALIGN",         (1,1), (-1,-1), "CENTER"),
    ]
    for i, row in enumerate(spc_table_data[1:], 1):
        c = statut_cell_color(row[2])
        spc_style.append(("TEXTCOLOR", (2,i), (2,i), c))
        spc_style.append(("FONTNAME",  (2,i), (2,i), "Helvetica-Bold"))
    spc_t.setStyle(TableStyle(spc_style))
    story.append(spc_t)

    # ════════════════════════════════════════════════════════════
    # PAGE 4 — AMDEC + PARETO + CONCLUSION
    # ════════════════════════════════════════════════════════════
    story.append(PageBreak())

    # AMDEC
    story += section_header("5. ANALYSE AMDEC — TOP RISQUES")

    fmea_df = df.copy()
    fmea_df["RPN"] = fmea_df["Severity"] * fmea_df["Occurrence"] * fmea_df["Detection"]
    fmea_top = fmea_df.nlargest(8, "RPN")[
        ["Part_ID","Defect_Type","Severity","Occurrence","Detection","RPN"]
    ].reset_index(drop=True)

    amdec_header = ["Reference", "Defaut", "G", "O", "D", "RPN", "Statut"]
    amdec_rows   = [amdec_header]
    for _, row in fmea_top.iterrows():
        rpn = int(row["RPN"])
        st_txt = "CRITIQUE" if rpn >= 150 else "ELEVE" if rpn >= 100 else "MOYEN"
        amdec_rows.append([
            str(row["Part_ID"])[:16],
            str(row["Defect_Type"])[:14],
            str(int(row["Severity"])),
            str(int(row["Occurrence"])),
            str(int(row["Detection"])),
            str(rpn),
            st_txt,
        ])

    amdec_t = Table(amdec_rows, colWidths=[3.5*cm, 3.2*cm, 1.5*cm, 1.5*cm, 1.5*cm, 2*cm, 3.8*cm])
    amdec_style_list = [
        ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#7c3aed")),
        ("TEXTCOLOR",     (0,0), (-1,0), BLANC),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,0), 9),
        ("ALIGN",         (0,0), (-1,0), "CENTER"),
        ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",      (0,1), (-1,-1), 8),
        ("TEXTCOLOR",     (0,1), (1,-1), GRIS_MED),
        ("TEXTCOLOR",     (2,1), (5,-1), BLANC),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [BLEU_MED, colors.HexColor("#0d1a2e")]),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("RIGHTPADDING",  (0,0), (-1,-1), 6),
        ("BOX",           (0,0), (-1,-1), 0.5, GRIS_DIM),
        ("ALIGN",         (2,1), (5,-1), "CENTER"),
        ("ALIGN",         (6,1), (6,-1), "CENTER"),
    ]
    for i, row in enumerate(amdec_rows[1:], 1):
        rpn_val = int(row[5])
        c = ROUGE if rpn_val >= 150 else ORANGE if rpn_val >= 100 else VERT
        amdec_style_list.append(("TEXTCOLOR", (5,i), (5,i), c))
        amdec_style_list.append(("FONTNAME",  (5,i), (5,i), "Helvetica-Bold"))
        amdec_style_list.append(("TEXTCOLOR", (6,i), (6,i), c))
        amdec_style_list.append(("FONTNAME",  (6,i), (6,i), "Helvetica-Bold"))

    amdec_t.setStyle(TableStyle(amdec_style_list))
    story.append(amdec_t)
    story.append(Spacer(1, 0.5*cm))

    # Pareto
    story += section_header("6. ANALYSE PARETO DES DEFAUTS")

    defects_df = df[df["Defect_Type"].astype(str).str.upper() != "OK"]
    if not defects_df.empty:
        pareto_counts = defects_df["Defect_Type"].value_counts()
        total_def = pareto_counts.sum()
        pareto_header = ["Type de defaut", "Nombre", "% du total", "Cumul %"]
        pareto_rows = [pareto_header]
        cumul = 0
        for defect, count in pareto_counts.items():
            pct   = count / total_def * 100
            cumul += pct
            pareto_rows.append([str(defect), str(count), f"{pct:.1f}%", f"{cumul:.1f}%"])

        par_t = Table(pareto_rows, colWidths=[6*cm, 3*cm, 4*cm, 4*cm])
        par_t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#0369a1")),
            ("TEXTCOLOR",     (0,0), (-1,0), BLANC),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,0), 9),
            ("ALIGN",         (0,0), (-1,0), "CENTER"),
            ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
            ("FONTSIZE",      (0,1), (-1,-1), 9),
            ("TEXTCOLOR",     (0,1), (0,-1), BLANC),
            ("TEXTCOLOR",     (1,1), (-1,-1), GRIS_MED),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [BLEU_MED, colors.HexColor("#0d1a2e")]),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
            ("RIGHTPADDING",  (0,0), (-1,-1), 10),
            ("BOX",           (0,0), (-1,-1), 0.5, GRIS_DIM),
            ("ALIGN",         (1,1), (-1,-1), "CENTER"),
        ]))
        story.append(par_t)
    else:
        story.append(Paragraph("✅ Aucun defaut detecte dans les donnees.", ST["body"]))

    story.append(Spacer(1, 0.6*cm))

    # CONCLUSION
    story += section_header("7. CONCLUSION ET RECOMMANDATIONS")

    if cpk >= 1.33:
        conclusion_txt = (
            f"Le processus est CAPABLE avec un Cpk = {cpk:.2f} (seuil requis : 1.33). "
            "Le systeme de mesure et le processus de production sont maitrisés. "
            "Maintenir la surveillance SPC et continuer les audits MSA periodiques."
        )
        conclusion_color = VERT
    elif cpk >= 1.0:
        conclusion_txt = (
            f"Le processus est LIMITE avec un Cpk = {cpk:.2f}. "
            "Une amelioration est necessaire pour atteindre le seuil de 1.33. "
            "Actions recommandees : recentrage du processus, reduction de la variabilite, "
            "verification du systeme de mesure (MSA)."
        )
        conclusion_color = ORANGE
    else:
        conclusion_txt = (
            f"Le processus est NON CAPABLE avec un Cpk = {cpk:.2f}. "
            "Des actions correctives immediates sont requises. "
            "Arreter la production si necessaire, analyser les causes racines (5M), "
            "mettre en place un plan d'action QRQC et surveiller l'efficacite des actions."
        )
        conclusion_color = ROUGE

    concl_data = [[Paragraph(conclusion_txt, ST["conclusion"])]]
    concl_table = Table(concl_data, colWidths=[17*cm])
    concl_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), BLEU_MED),
        ("BOX",           (0,0), (-1,-1), 2, conclusion_color),
        ("LEFTBORDER",    (0,0), (0,-1), 5, conclusion_color),
        ("TOPPADDING",    (0,0), (-1,-1), 14),
        ("BOTTOMPADDING", (0,0), (-1,-1), 14),
        ("LEFTPADDING",   (0,0), (-1,-1), 16),
        ("RIGHTPADDING",  (0,0), (-1,-1), 16),
    ]))
    story.append(concl_table)
    story.append(Spacer(1, 0.6*cm))

    # Signature
    sign_data = [
        [
            Paragraph("Redige par :", ST["small"]),
            Paragraph("Verifie par :", ST["small"]),
            Paragraph("Approuve par :", ST["small"]),
        ],
        [
            Paragraph("_______________________", ST["body"]),
            Paragraph("_______________________", ST["body"]),
            Paragraph("_______________________", ST["body"]),
        ],
        [
            Paragraph(f"Date : {datetime.now().strftime('%d/%m/%Y')}", ST["small"]),
            Paragraph("Date : ___/___/______", ST["small"]),
            Paragraph("Date : ___/___/______", ST["small"]),
        ],
    ]
    sign_table = Table(sign_data, colWidths=[5.5*cm, 5.5*cm, 5.5*cm])
    sign_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), BLEU_MED),
        ("BOX",           (0,0), (-1,-1), 0.5, GRIS_DIM),
        ("INNERGRID",     (0,0), (-1,-1), 0.3, colors.HexColor("#1e293b")),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(sign_table)

    # ─── BUILD ──────────────────────────────────────────────────
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)

    buffer.seek(0)
    with open(PDF_PATH, "wb") as f:
        f.write(buffer.read())

    return PDF_PATH

# =========================
# LAYOUT
# =========================
def render_sidebar(metrics: dict) -> str:
    with st.sidebar:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=170)
        else:
            st.markdown(f"## {APP_NAME}")

        st.markdown("### MENU PRINCIPAL")
        page = st.radio("", MENU_ITEMS, label_visibility="collapsed", key="main_menu")
        st.markdown("---")
        st.markdown("### 📌 Indicateurs")
        st.metric("Total mesures", metrics["total"])
        st.metric("Points MSA", metrics["msa_count"])
        st.metric("Points SPC", metrics["spc_count"])
        st.caption(f"🕐 Dernière MAJ : {datetime.now().strftime('%H:%M:%S')}")

    return clean_page_name(page)


# =========================
# PAGES
# =========================
def page_saisie_mesures(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("➕ Saisie des mesures")

    with st.form("form_mesures"):
        col1, col2, col3 = st.columns(3)

        with col1:
            data_type = st.selectbox("Type de données", ["SPC", "MSA"])
            part_id = st.text_input("Référence / Part ID")
            operator = st.text_input("Opérateur")

        with col1:
            machine = st.text_input("Machine", value="M1")
            usl = st.number_input("USL", value=12.1000, format="%.4f")
            lsl = st.number_input("LSL", value=11.9000, format="%.4f")

        with col1:
            mesure_1 = st.number_input("Mesure 1", format="%.4f")
            mesure_2 = st.number_input("Mesure 2", format="%.4f")
            mesure_3 = st.number_input("Mesure 3", format="%.4f")

        submitted = st.form_submit_button("Enregistrer")
    if submitted:
        part_id_final = f"{data_type}_{part_id}"

        new_rows = pd.DataFrame([
            {
               "Date_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Part_ID": part_id_final,
                "Operator": operator,
                "Trial": 1,
                "Measurement": mesure_1,
                "USL": usl,
                "LSL": lsl,
                "Machine": machine,
                "Defect_Type": "OK",
                "Severity": 1,
                "Occurrence": 1,
                "Detection": 1,
            },
            {
                "Date_Time": datetime.now(),
                "Part_ID": part_id_final,
                "Operator": operator,
                "Trial": 2,
                "Measurement": mesure_2,
                "USL": usl,
                "LSL": lsl,
                "Machine": machine,
                "Defect_Type": "OK",
                "Severity": 1,
                "Occurrence": 1,
                "Detection": 1,
            },
            {
                "Date_Time": datetime.now(),
                "Part_ID": part_id_final,
                "Operator": operator,
                "Trial": 3,
                "Measurement": mesure_3,
                "USL": usl,
                "LSL": lsl,
                "Machine": machine,
                "Defect_Type": "OK",
                "Severity": 1,
                "Occurrence": 1,
                "Detection": 1,
            },
        ])

        for _, row in new_rows.iterrows():
            save_to_google_sheet(row.to_dict())

        st.session_state["manual_data"] = pd.concat(
            [st.session_state.get("manual_data", pd.DataFrame()), new_rows],
            ignore_index=True,
        )

        st.success("✅ Mesures enregistrées")

    if "manual_data" in st.session_state:
        df = pd.concat([df, st.session_state["manual_data"]], ignore_index=True)

    return df


def page_dashboard(df: pd.DataFrame, metrics: dict) -> None:
    st.subheader("🏠 Vue générale")
    col_a, col_b = st.columns(2)

    with col_a:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(df) + 1)),
                y=df["Measurement"],
                mode="lines+markers",
                name="Mesures",
            )
        )
        fig.add_hline(y=metrics["mean_val"], line_dash="dash", annotation_text="Moyenne")
        fig.add_hline(y=metrics["usl"], line_dash="dot", annotation_text="USL")
        fig.add_hline(y=metrics["lsl"], line_dash="dot", annotation_text="LSL")
        fig.update_layout(title="Évolution des mesures", template="plotly_dark", height=420)
        plot_chart(fig, "dashboard_evolution")

    with col_b:
        fig = px.histogram(df, x="Measurement", nbins=25, template="plotly_dark", title="Distribution des mesures")
        fig.add_vline(x=metrics["usl"], line_dash="dash", annotation_text="USL")
        fig.add_vline(x=metrics["lsl"], line_dash="dash", annotation_text="LSL")
        fig.add_vline(x=metrics["mean_val"], line_dash="dot", annotation_text="Moyenne")
        plot_chart(fig, "dashboard_distribution")

    context = f"""
Moyenne = {metrics['mean_val']:.4f}
Écart-type = {metrics['std_val']:.6f}
Cp = {metrics['cp']:.2f}
Cpk = {metrics['cpk']:.2f}
USL = {metrics['usl']:.4f}
LSL = {metrics['lsl']:.4f}
Nombre total de mesures = {metrics['total']}
"""
    show_ai_analysis("Tableau de bord", context)


def page_msa(df: pd.DataFrame, metrics: dict) -> None:
    st.subheader("📏 Module MSA complet")
    msa_data = metrics["msa_data"]
    usl = metrics["usl"]
    lsl = metrics["lsl"]

    tab_summary, tab_msa1, tab_grr, tab_bias, tab_stability, tab_linearity, tab_attribute = st.tabs(
        ["Résumé", "Type 1", "Gage R&R", "Bias", "Stability", "Linearity", "Attribute MSA"]
    )

    with tab_summary:
        st.markdown("### 📌 Résumé MSA")
        st.info("MSA sert à vérifier si le système de mesure est fiable avant de juger le processus.")

        if msa_data.empty:
            st.warning("Aucune donnée MSA détectée. Ajoute des lignes avec Part_ID contenant MSA.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Mesures MSA", len(msa_data))
            c2.metric("Opérateurs", msa_data["Operator"].nunique())
            c3.metric("Pièces", msa_data["Part_ID"].nunique())

        st.markdown("""
**Pourquoi cette étape est importante ?**
- **Type 1** : vérifie la répétabilité d’un seul moyen de mesure.
- **Gage R&R** : sépare la variation appareil / opérateur / pièce.
- **Bias** : compare la mesure à une valeur de référence.
- **Stability** : vérifie si le système dérive dans le temps.
- **Linearity** : vérifie si le biais change selon le niveau de mesure.
- **Attribute MSA** : utile pour les décisions OK / NOK.
""")

    with tab_msa1:
        st.markdown("### 📏 MSA Type 1")

        if msa_data.empty:
            st.warning("Aucune donnée MSA disponible.")
        else:
            mean_msa = float(msa_data["Measurement"].mean())
            std_msa = safe_std(msa_data["Measurement"])
            ref = (usl + lsl) / 2
            tolerance = usl - lsl
            cg = (0.2 * tolerance) / (6 * std_msa) if std_msa > 0 else 0
            cgk = (0.1 * tolerance - abs(mean_msa - ref)) / (3 * std_msa) if std_msa > 0 else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Référence", f"{ref:.4f}")
            c2.metric("Tolérance", f"{tolerance:.4f}")
            c3.metric("Cg", f"{cg:.2f}")
            c4.metric("Cgk", f"{cgk:.2f}")

            if cgk < 1:
                st.error("❌ Système de mesure NON acceptable (Cgk < 1)")
            elif cgk < 1.33:
                st.warning("⚠️ Système limite (amélioration recommandée)")
            else:
                st.success("✅ Système de mesure acceptable")

            st.markdown("""
**Lecture rapide:**
- Cg ≥ 1.33 → répétabilité correcte
- Cgk ≥ 1.33 → système fiable
- Cgk < 1 → système NON fiable
""")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(msa_data) + 1)),
                    y=msa_data["Measurement"],
                    mode="lines+markers",
                    name="Mesures MSA",
                )
            )
            fig.add_hline(y=mean_msa, line_dash="dash", annotation_text="Moyenne")
            fig.add_hline(y=ref, line_dash="dot", annotation_text="Référence")
            fig.update_layout(title="Carte MSA Type 1", template="plotly_dark", height=430)
            plot_chart(fig, "msa_type1_chart")

        context = f"""
Référence = {ref:.4f}
Tolérance = {tolerance:.4f}
Moyenne MSA = {mean_msa:.4f}
Écart-type MSA = {std_msa:.6f}
Cg = {cg:.2f}
Cgk = {cgk:.2f}
Nombre de mesures MSA = {len(msa_data)}
"""
        show_ai_analysis("MSA Type 1", context)

    with tab_grr:
        st.markdown("### ⚙️ Gage R&R")
        if msa_data.empty:
            st.warning("Aucune donnée MSA disponible.")
        else:
            df_grr = msa_data.copy()
            var_total = df_grr["Measurement"].var()
            var_operator = df_grr.groupby("Operator")["Measurement"].mean().var()
            var_repeat = df_grr.groupby(["Part_ID", "Operator"])["Measurement"].var().mean()
            var_total = 0 if pd.isna(var_total) else var_total
            var_operator = 0 if pd.isna(var_operator) else var_operator
            var_repeat = 0 if pd.isna(var_repeat) else var_repeat
            var_grr = var_operator + var_repeat
            percent_grr = (var_grr / var_total) * 100 if var_total > 0 else 0

            c1, c2, c3 = st.columns(3)
            c1.metric("Variation totale", f"{var_total:.8f}")
            c2.metric("GRR", f"{var_grr:.8f}")
            c3.metric("%GRR", f"{percent_grr:.2f}%")

            fig = px.box(df_grr, x="Operator", y="Measurement", color="Operator", title="Variation par opérateur", template="plotly_dark")
            plot_chart(fig, "msa_grr_box")

            context = f"""
Variation totale = {var_total:.8f}
Variation opérateur = {var_operator:.8f}
Variation répétabilité = {var_repeat:.8f}
GRR = {var_grr:.8f}
%GRR = {percent_grr:.2f}
"""
            show_ai_analysis("Gage R&R", context)

    with tab_bias:
        st.markdown("### 🎯 Bias")
        if msa_data.empty:
            st.warning("Aucune donnée MSA disponible.")
        else:
            reference = st.number_input("Valeur de référence", value=12.0000, format="%.4f", key="bias_reference")
            mean_bias = float(msa_data["Measurement"].mean())
            bias = mean_bias - reference
            c1, c2 = st.columns(2)
            c1.metric("Moyenne mesurée", f"{mean_bias:.6f}")
            c2.metric("Bias", f"{bias:.6f}")
            context = f"""
Référence = {reference:.4f}
Moyenne mesurée = {mean_bias:.6f}
Bias = {bias:.6f}
"""
            show_ai_analysis("Bias", context)

    with tab_linearity:
        st.markdown("### 📈 Linearity")
        if msa_data.empty:
            st.warning("Aucune donnée MSA disponible.")
        else:
            df_lin = msa_data.copy()
            df_lin["Reference"] = df_lin.groupby("Part_ID")["Measurement"].transform("mean")
            df_lin["Bias"] = df_lin["Measurement"] - df_lin["Reference"]
            fig = px.scatter(df_lin, x="Reference", y="Bias", color="Operator", title="Linearity : Bias vs Référence", template="plotly_dark")
            plot_chart(fig, "msa_linearity_scatter")
            bias_var = safe_std(df_lin.groupby("Part_ID")["Bias"].mean())
            st.metric("Variation du Bias", f"{bias_var:.6f}")
            context = f"""
Variation du Bias = {bias_var:.6f}
Nombre de pièces MSA = {df_lin['Part_ID'].nunique()}
Nombre opérateurs = {df_lin['Operator'].nunique()}
"""
            show_ai_analysis("Linearity", context)

    with tab_stability:
        st.markdown("### ⏳ Stability")
        if msa_data.empty:
            st.warning("Aucune donnée MSA disponible.")
        else:
            df_stab = msa_data.copy()
            df_stab["Date_Time"] = pd.to_datetime(df_stab["Date_Time"], errors="coerce")
            df_stab = df_stab.dropna(subset=["Date_Time"]).sort_values("Date_Time")
            mean_stab = float(df_stab["Measurement"].mean())
            std_stab = safe_std(df_stab["Measurement"])
            ucl_stab = mean_stab + 3 * std_stab
            lcl_stab = mean_stab - 3 * std_stab
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_stab["Date_Time"], y=df_stab["Measurement"], mode="lines+markers", name="Mesures"))
            fig.add_hline(y=mean_stab, line_dash="dash", annotation_text="Moyenne")
            fig.add_hline(y=ucl_stab, line_dash="dot", annotation_text="UCL")
            fig.add_hline(y=lcl_stab, line_dash="dot", annotation_text="LCL")
            fig.update_layout(title="Stability dans le temps", template="plotly_dark", height=430)
            plot_chart(fig, "msa_stability_chart")
            out_stab = df_stab[(df_stab["Measurement"] > ucl_stab) | (df_stab["Measurement"] < lcl_stab)]
            context = f"""
Moyenne stabilité = {mean_stab:.4f}
Écart-type stabilité = {std_stab:.6f}
UCL = {ucl_stab:.4f}
LCL = {lcl_stab:.4f}
Points instables = {len(out_stab)}
"""
            show_ai_analysis("Stability", context)

    with tab_attribute:
        st.markdown("### ✅ Attribute MSA")
        df_attr = df.copy()
        df_attr["Decision"] = df_attr["Defect_Type"].astype(str).str.upper().apply(lambda x: "OK" if x == "OK" else "NOK")
        ok_count = len(df_attr[df_attr["Decision"] == "OK"])
        nok_count = len(df_attr[df_attr["Decision"] == "NOK"])
        agreement = (ok_count / len(df_attr)) * 100 if len(df_attr) > 0 else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("OK", ok_count)
        c2.metric("NOK", nok_count)
        c3.metric("% OK", f"{agreement:.2f}%")
        fig = px.pie(df_attr, names="Decision", title="Répartition OK / NOK", template="plotly_dark")
        plot_chart(fig, "msa_attribute_pie")
        context = f"""
OK = {ok_count}
NOK = {nok_count}
%OK = {agreement:.2f}
Nombre total = {len(df_attr)}
"""
        show_ai_analysis("Attribute MSA", context)


def page_spc(metrics: dict) -> None:
    st.subheader("📉 Module SPC complet")
    spc_data = metrics["spc_data"]
    usl = metrics["usl"]
    lsl = metrics["lsl"]

    mean_spc = float(spc_data["Measurement"].mean())
    std_spc = safe_std(spc_data["Measurement"])
    ucl = mean_spc + 3 * std_spc
    lcl = mean_spc - 3 * std_spc

    spc_work = spc_data.copy().reset_index(drop=True)
    spc_work["Point"] = range(1, len(spc_work) + 1)
    spc_work["Hors_Controle"] = (spc_work["Measurement"] > ucl) | (spc_work["Measurement"] < lcl)

    tab_control, tab_rules, tab_distribution, tab_machine, tab_ai = st.tabs(
        ["Carte de contrôle", "Règles SPC", "Distribution", "Machine / Opérateur", "Interprétation IA"]
    )

    with tab_control:
        st.markdown("### 📈 Carte de contrôle")
        c1, c2, c3 = st.columns(3)
        c1.metric("CL", f"{mean_spc:.4f}")
        c2.metric("UCL", f"{ucl:.4f}")
        c3.metric("LCL", f"{lcl:.4f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spc_work["Point"], y=spc_work["Measurement"], mode="lines+markers", name="Mesures"))
        fig.add_hline(y=mean_spc, line_dash="dash", annotation_text="CL")
        fig.add_hline(y=ucl, line_dash="dash", annotation_text="UCL")
        fig.add_hline(y=lcl, line_dash="dash", annotation_text="LCL")
        fig.add_hline(y=usl, line_dash="dot", annotation_text="USL")
        fig.add_hline(y=lsl, line_dash="dot", annotation_text="LSL")
        fig.update_layout(title="Carte de contrôle SPC", template="plotly_dark", height=460)
        plot_chart(fig, "spc_control_chart")

        out_control = spc_work[spc_work["Hors_Controle"]]
        if not out_control.empty:
            st.error(f"❌ {len(out_control)} point(s) hors contrôle")
            st.dataframe(out_control, use_container_width=True, hide_index=True)
        else:
            st.success("✅ Aucun point hors contrôle")

    with tab_rules:
        st.markdown("### 🚦 Règles SPC")
        out_control = spc_work[spc_work["Hors_Controle"]]
        rule1 = len(out_control)

        values = spc_work["Measurement"].dropna().tolist()
        rule2 = False
        if len(values) >= 7:
            above = [v > mean_spc for v in values]
            current_run = 1
            max_run = 1
            for i in range(1, len(above)):
                if above[i] == above[i - 1]:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 1
            rule2 = max_run >= 7

        trend_detected = False
        if len(values) >= 6:
            for i in range(len(values) - 5):
                segment = values[i : i + 6]
                increasing = all(segment[j] < segment[j + 1] for j in range(5))
                decreasing = all(segment[j] > segment[j + 1] for j in range(5))
                if increasing or decreasing:
                    trend_detected = True
                    break

        r1, r2, r3 = st.columns(3)
        r1.error(f"❌ {rule1} point(s) hors contrôle") if rule1 > 0 else r1.success("✅ Règle 1 OK")
        r2.warning("⚠️ 7 points du même côté") if rule2 else r2.success("✅ Règle 2 OK")
        r3.warning("⚠️ Tendance détectée") if trend_detected else r3.success("✅ Règle 3 OK")

    with tab_distribution:
        st.markdown("### 📊 Distribution")
        fig = px.histogram(spc_work, x="Measurement", nbins=25, title="Histogramme SPC", template="plotly_dark")
        fig.add_vline(x=mean_spc, line_dash="dot", annotation_text="Moyenne")
        fig.add_vline(x=usl, line_dash="dash", annotation_text="USL")
        fig.add_vline(x=lsl, line_dash="dash", annotation_text="LSL")
        plot_chart(fig, "spc_distribution_hist")


    with tab_machine:
        st.markdown("### 🏭 Machine / Opérateur")
        col_m, col_o = st.columns(2)

        with col_m:
            if "Machine" in spc_work.columns:
                machine_stats = spc_work.groupby("Machine")["Measurement"].agg(["count", "mean", "std"]).reset_index()
                machine_stats.columns = ["Machine", "Nombre", "Moyenne", "Écart-type"]
                st.dataframe(machine_stats, use_container_width=True, hide_index=True)
                fig = px.box(spc_work, x="Machine", y="Measurement", color="Machine", template="plotly_dark", title="Variation par machine")
                plot_chart(fig, "spc_machine_box")
            else:
                st.warning("Colonne Machine introuvable")

        with col_o:
            if "Operator" in spc_work.columns:
                operator_stats = spc_work.groupby("Operator")["Measurement"].agg(["count", "mean", "std"]).reset_index()
                operator_stats.columns = ["Opérateur", "Nombre", "Moyenne", "Écart-type"]
                st.dataframe(operator_stats, use_container_width=True, hide_index=True)
                fig = px.box(spc_work, x="Operator", y="Measurement", color="Operator", template="plotly_dark", title="Variation par opérateur")
                plot_chart(fig, "spc_operator_box")
            else:
                st.warning("Colonne Operator introuvable")

    with tab_ai:
        context = f"""
Moyenne SPC = {mean_spc:.4f}
Écart-type SPC = {std_spc:.6f}
UCL = {ucl:.4f}
LCL = {lcl:.4f}
Points hors contrôle = {len(spc_work[spc_work['Hors_Controle']])}
Cp = {metrics['cp']:.2f}
Cpk = {metrics['cpk']:.2f}
"""
        show_ai_analysis("SPC", context)



def page_capability(df: pd.DataFrame, metrics: dict) -> None:
    st.subheader("🎯 Module Capabilité complet")
    tab_kpi, tab_hist, tab_centering, tab_machine, tab_ai = st.tabs(
        ["Indices Cp / Cpk", "Histogramme", "Centrage", "Machine / Opérateur", "Interprétation IA"]
    )

    with tab_kpi:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("LS", f"{metrics['usl']:.4f}")
        c2.metric("LI", f"{metrics['lsl']:.4f}")
        c3.metric("Cp", f"{metrics['cp']:.2f}")
        c4.metric("Cpk", f"{metrics['cpk']:.2f}")
        process_status(metrics["cpk"])

    with tab_hist:
        fig = px.histogram(df, x="Measurement", nbins=25, title="Distribution des mesures", template="plotly_dark")
        fig.add_vline(x=metrics["usl"], line_dash="dash", annotation_text="LS")
        fig.add_vline(x=metrics["lsl"], line_dash="dash", annotation_text="LI")
        fig.add_vline(x=metrics["mean_val"], line_dash="dot", annotation_text="Moyenne")
        plot_chart(fig, "cap_hist")

    with tab_centering:
        target = (metrics["usl"] + metrics["lsl"]) / 2
        decentrage = metrics["mean_val"] - target
        c1, c2, c3 = st.columns(3)
        c1.metric("Cible", f"{target:.4f}")
        c2.metric("Moyenne", f"{metrics['mean_val']:.4f}")
        c3.metric("Décalage", f"{decentrage:.6f}")

    with tab_machine:
        col_m, col_o = st.columns(2)
        with col_m:
            machine_stats = df.groupby("Machine")["Measurement"].agg(["count", "mean", "std"]).reset_index()
            machine_stats.columns = ["Machine", "Nombre", "Moyenne", "Écart-type"]
            st.dataframe(machine_stats, use_container_width=True, hide_index=True)
            fig = px.box(df, x="Machine", y="Measurement", color="Machine", template="plotly_dark", title="Distribution par machine")
            plot_chart(fig, "cap_machine_box")
        with col_o:
            operator_stats = df.groupby("Operator")["Measurement"].agg(["count", "mean", "std"]).reset_index()
            operator_stats.columns = ["Opérateur", "Nombre", "Moyenne", "Écart-type"]
            st.dataframe(operator_stats, use_container_width=True, hide_index=True)
            fig = px.box(df, x="Operator", y="Measurement", color="Operator", template="plotly_dark", title="Distribution par opérateur")
            plot_chart(fig, "cap_operator_box")

    with tab_ai:
        target = (metrics["usl"] + metrics["lsl"]) / 2
        decentrage = metrics["mean_val"] - target
        context = f"""
Moyenne = {metrics['mean_val']:.4f}
Écart-type = {metrics['std_val']:.6f}
LS = {metrics['usl']:.4f}
LI = {metrics['lsl']:.4f}
Cible = {target:.4f}
Décalage = {decentrage:.6f}
Cp = {metrics['cp']:.2f}
Cpk = {metrics['cpk']:.2f}
Nombre de mesures = {metrics['total']}
"""
      st.markdown("---")
    st.subheader("📄 Télécharger Rapport Qualité")
    
    if st.button("📄 Générer Rapport PDF", key="cap_pdf_btn", type="primary", use_container_width=True):
        try:
            with st.spinner("Génération du PDF..."):
                pdf_path = generate_pdf_report(metrics, df)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "⬇ Télécharger Rapport PDF",
                    f,
                    f"Rapport_Qualite_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    "application/pdf",
                    key="cap_pdf_dl",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Erreur PDF: {e}")
    
    st.markdown("---")
    show_ai_analysis("Capabilité", context)
def page_pareto(df: pd.DataFrame) -> None:
    st.subheader("📊 Analyse Pareto des défauts")
    defects = df[df["Defect_Type"].astype(str).str.upper() != "OK"]

    if defects.empty:
        st.success("✅ Aucun défaut détecté.")
        return

    pareto = defects["Defect_Type"].value_counts().reset_index()
    pareto.columns = ["Type de défaut", "Nombre"]
    pareto["Cumul %"] = pareto["Nombre"].cumsum() / pareto["Nombre"].sum() * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(x=pareto["Type de défaut"], y=pareto["Nombre"], name="Défauts"))
    fig.add_trace(go.Scatter(x=pareto["Type de défaut"], y=pareto["Cumul %"], yaxis="y2", mode="lines+markers", name="Cumul %"))
    fig.update_layout(
        title="Diagramme Pareto",
        template="plotly_dark",
        height=460,
        yaxis=dict(title="Nombre"),
        yaxis2=dict(title="Cumul %", overlaying="y", side="right", range=[0, 110]),
    )
    plot_chart(fig, "pareto_chart")
    st.dataframe(pareto, use_container_width=True, hide_index=True)

    top_defect = pareto.iloc[0]["Type de défaut"]
    top_count = pareto.iloc[0]["Nombre"]
    context = f"""
Défaut principal = {top_defect}
Occurrences défaut principal = {top_count}
Nombre total défauts = {len(defects)}
"""
    show_ai_analysis("Pareto", context)


def page_amdec(df: pd.DataFrame) -> None:
    st.subheader("⚠️ Analyse AMDEC automatique")
    fmea = df.copy()
    fmea["RPN"] = fmea["Severity"] * fmea["Occurrence"] * fmea["Detection"]

    def get_status(rpn: float) -> str:
        if rpn >= 150:
            return "🔴 Critique"
        if rpn >= 100:
            return "🟡 Élevé"
        return "🟢 Moyen"

    def get_action(rpn: float) -> str:
        if rpn >= 150:
            return "Action immédiate requise"
        if rpn >= 100:
            return "Amélioration nécessaire"
        return "Risque acceptable"

    fmea["Statut"] = fmea["RPN"].apply(get_status)
    fmea["Action"] = fmea["RPN"].apply(get_action)
    fmea = fmea.sort_values(by="RPN", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("RPN maximum", int(fmea["RPN"].max()))
    c2.metric("RPN moyen", f"{fmea['RPN'].mean():.1f}")
    c3.metric("Risques critiques", len(fmea[fmea["RPN"] >= 150]))

    table_fmea = fmea[["Part_ID", "Defect_Type", "Severity", "Occurrence", "Detection", "RPN", "Statut", "Action"]].rename(
        columns={
            "Part_ID": "Référence pièce",
            "Defect_Type": "Type de défaut",
            "Severity": "Gravité",
            "Detection": "Détection",
        }
    )
    st.dataframe(table_fmea, use_container_width=True, hide_index=True)

    top_risk = fmea.iloc[0]
    context = f"""
RPN maximum = {int(fmea['RPN'].max())}
Défaut principal = {top_risk['Defect_Type']}
Gravité = {top_risk['Severity']}
Occurrence = {top_risk['Occurrence']}
Détection = {top_risk['Detection']}
Statut = {top_risk['Statut']}
Action actuelle = {top_risk['Action']}
"""
    show_ai_analysis("AMDEC", context)


def page_ai(metrics: dict) -> None:
    st.subheader("🤖 Assistant Qualité IA")
    question = st.text_area("Pose ta question qualité", key="ai_question")

    if st.button("Analyser", key="ai_analyze_button"):
        if not question.strip():
            st.warning("Écris une question")
            return

    prompt = f"""
Tu es un expert qualité automobile.

Données actuelles :
- Moyenne = {metrics['mean_val']:.4f}
- Écart-type = {metrics['std_val']:.6f}
- Cp = {metrics['cp']:.2f}
- Cpk = {metrics['cpk']:.2f}
- USL = {metrics['usl']}
- LSL = {metrics['lsl']}
- Nombre de mesures = {metrics['total']}

Question :
{question}

Donne :
1. Interprétation
2. Causes possibles
3. Actions immédiates
4. Actions correctives
"""

    with st.spinner("🤖 Analyse en cours..."):
        answer = ask_hf_ai(prompt)

    st.markdown("### 🧠 Réponse IA")
    st.success(answer)

def render_pdf_section(metrics: dict) -> None:
    st.markdown("---")
    st.subheader("📄 Rapport Qualité")

    if st.button("Générer le rapport PDF", key="generate_pdf_button"):
        pdf_path = generate_pdf_report(metrics, df)

        with open(pdf_path, "rb") as file:
            st.download_button(
                label="📥 Télécharger le rapport PDF",
                data=file,
                file_name="rapport_qualite_specsense.pdf",
                mime="application/pdf",
                key="download_pdf_button",
            )
def render_footer() -> None:
    st.markdown("---")
    st.caption(
        f"{APP_NAME} {APP_VERSION} | Qualité 4.0 | Inspiré IATF 16949"
    )


def render_header() -> None:
    h1, h2 = st.columns([1, 5])

    with h1:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=120)

    with h2:
        st.markdown(
            f"""
            <div style="
                padding:25px;
                border-radius:22px;
                background:linear-gradient(135deg,#0f172a,#1e293b);
                border:1px solid rgba(255,255,255,0.08);
            ">
                <h1 style="margin:0; font-size:42px; font-weight:900; color:white;">
                    {APP_NAME}
                </h1>
                <p style="margin-top:10px; font-size:18px; color:#94a3b8;">
                    Plateforme intelligente de qualité industrielle
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)


def render_global_kpis(metrics: dict) -> None:
    st.markdown("### 📊 KPIs Globaux")

    col1, col2, col3, col4 = st.columns(4)

    total_rows = metrics.get("total", 0)
    msa_count = metrics.get("msa_count", 0)
    spc_count = metrics.get("spc_count", 0)
    avg_value = metrics.get("mean_val", 0)

    col1.metric("Mesures", total_rows)
    col2.metric("MSA", msa_count)
    col3.metric("SPC", spc_count)
    col4.metric("Moyenne", f"{avg_value:.2f}")

    st.markdown("<br>", unsafe_allow_html=True)
# =========================
# MAIN
# =========================
def main() -> None:
    inject_css()

    try:
        df = load_data()
    except Exception as exc:
        st.error("🚨 Impossible de lire Google Sheet.")
        st.write(exc)
        return

    if df.empty and "manual_data" not in st.session_state:
        st.warning("⚠️ Google Sheet vide — commencez par saisir des données.")
        page_saisie_mesures(df)
        return

    if "manual_data" in st.session_state:
        df = pd.concat([df, st.session_state["manual_data"]], ignore_index=True)

    metrics = prepare_data(df)

    page = render_sidebar(metrics)

    render_header()
    render_global_kpis(metrics)

    if page == "Saisie Mesures":
        df = page_saisie_mesures(df)
        metrics = prepare_data(df)

    elif page == "Tableau de bord":
        page_dashboard(df, metrics)

    elif page == "MSA":
        page_msa(df, metrics)

    elif page == "SPC":
        page_spc(metrics)

    elif page == "Capabilité":
        page_capability(df, metrics)

    elif page == "Pareto":
        page_pareto(df)

    elif page == "AMDEC":
        page_amdec(df)

    elif page == "IA":
        page_ai(metrics)

    render_footer()


if __name__ == "__main__":
    main()
