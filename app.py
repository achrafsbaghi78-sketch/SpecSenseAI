import os
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from huggingface_hub import InferenceClient

st.set_page_config(
    page_title="SpecSense AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

def ask_hf_ai(question):
    if "HUGGINGFACE_TOKEN" not in st.secrets:
        return "❌ HUGGINGFACE_TOKEN manquant dans Streamlit Secrets."

    try:
        client = InferenceClient(token=st.secrets["HUGGINGFACE_TOKEN"])

        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un expert en qualité industrielle. Réponds en français simple avec des actions concrètes."
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            max_tokens=500,
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Erreur IA : {e}"

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #020617 0%, #07111f 45%, #0b1220 100%);
    color: white;
}
.block-container {
    padding-top: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #07111f 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}
div[data-testid="stMetric"] {
    background: radial-gradient(circle at top left, rgba(0,212,255,0.18), rgba(15,23,42,0.92));
    border: 1px solid rgba(0,212,255,0.45);
    border-radius: 24px;
    padding: 25px;
    box-shadow: 0 18px 40px rgba(0,0,0,0.35);
}
div[data-testid="stMetricLabel"] p {
    color: #cbd5e1 !important;
    font-weight: 700 !important;
    font-size: 16px !important;
}
div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 38px !important;
    font-weight: 900 !important;
}
div[role="radiogroup"] label {
    background: rgba(15,23,42,0.65);
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 14px;
    padding: 12px 14px;
    margin-bottom: 8px;
    transition: all 0.25s ease;
}
div[role="radiogroup"] label:hover {
    background: rgba(0,212,255,0.13);
    border-color: rgba(0,212,255,0.35);
    transform: translateX(4px);
}
@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 34px !important;
    }
}
</style>
""", unsafe_allow_html=True)

G_SHEET_URL = "https://docs.google.com/spreadsheets/d/1Xy4tgkGs1OXOTh-OMAsR7YsfkUPxttF7qalhDdhHa90/export?format=csv&gid=0"

@st.cache_data(ttl=60)
def load_data():
    df = pd.read_csv(G_SHEET_URL)
    df.columns = df.columns.str.strip()

    required_cols = [
        "Date_Time", "Part_ID", "Operator", "Trial",
        "Measurement", "USL", "LSL",
        "Machine", "Defect_Type",
        "Severity", "Occurrence", "Detection"
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Colonnes manquantes : {missing_cols}")
        st.stop()

    numeric_cols = ["Measurement", "USL", "LSL", "Severity", "Occurrence", "Detection"]

    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(",", ".", regex=False).str.strip()
        df[col]
