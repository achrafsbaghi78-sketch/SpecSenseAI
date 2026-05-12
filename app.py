"""
SpecSense AI - Plateforme Premium de Gestion de la Qualité
Design Luxe Professionnel - Production Ready
"""
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sqlite3
from datetime import datetime
import numpy as np

# ========================
# CONFIGURATION PAGE
# ========================
st.set_page_config(
    page_title="SpecSense AI - Qualité Intelligente",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========================
# CSS LUXE PREMIUM - SANS EMOJI
# ========================
def inject_css_premium():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html, body, .stApp {
        font-family: 'Poppins', 'Inter', sans-serif;
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1229 100%) !important;
        color: #e2e8f0;
        overflow-x: hidden;
    }
    
    .stApp {
        background-attachment: fixed;
    }
    
    /* ========================
       HEADER PREMIUM
       ======================== */
    .header-premium {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 50%, #1a1f3a 100%);
        padding: 50px 40px;
        border-radius: 24px;
        border: 1px solid rgba(59, 130, 246, 0.2);
        margin-bottom: 40px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .header-premium h1 {
        background: linear-gradient(135deg, #38bdf8 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 52px;
        font-weight: 900;
        margin: 0;
        letter-spacing: -2px;
    }
    
    .header-premium p {
        color: #94a3b8;
        font-size: 18px;
        margin: 15px 0 0 0;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    /* ========================
       SIDEBAR PREMIUM
       ======================== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1729 0%, #1a1f3a 100%) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.1);
        box-shadow: 2px 0 20px rgba(0, 0, 0, 0.3);
    }
    
    .sidebar-title {
        background: linear-gradient(135deg, #38bdf8 0%, #06b6d4 100%);
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 30px;
        border: 1px solid rgba(56, 189, 248, 0.2);
        box-shadow: 0 8px 32px rgba(56, 189, 248, 0.15);
    }
    
    .sidebar-title h2 {
        color: white;
        font-size: 24px;
        font-weight: 900;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    /* ========================
       METRICS PREMIUM
       ======================== */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(30, 41, 59, 0.6));
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(56, 189, 248, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    div[data-testid="stMetric"]:hover {
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 16px 48px rgba(56, 189, 248, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transform: translateY(-6px);
    }
    
    div[data-testid="stMetricLabel"] p {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin: 0 !important;
    }
    
    div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-size: 36px !important;
        font-weight: 900 !important;
        margin-top: 12px !important;
        text-shadow: 0 0 20px rgba(56, 189, 248, 0.3);
    }
    
    /* ========================
       BUTTONS PREMIUM
       ======================== */
    .stButton > button {
        background: linear-gradient(135deg, #38bdf8 0%, #06b6d4 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 40px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        letter-spacing: 0.7px !important;
        box-shadow: 0 10px 30px rgba(56, 189, 248, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase;
        cursor: pointer !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #06b6d4 0%, #38bdf8 100%) !important;
        box-shadow: 0 15px 40px rgba(56, 189, 248, 0.5) !important;
        transform: translateY(-3px) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 25px rgba(56, 189, 248, 0.4) !important;
    }
    
    /* BUTTON SECONDARY */
    .button-secondary {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.2), rgba(6, 182, 212, 0.15)) !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        color: #38bdf8 !important;
    }
    
    .button-secondary:hover {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.3), rgba(6, 182, 212, 0.25)) !important;
    }
    
    /* ========================
       TABS PREMIUM
       ======================== */
    div[role="tablist"] {
        background: linear-gradient(90deg, rgba(56, 189, 248, 0.05), transparent);
        padding: 16px;
        border-radius: 14px;
        border-bottom: 2px solid rgba(56, 189, 248, 0.2);
        gap: 10px;
    }
    
    button[role="tab"] {
        background: rgba(56, 189, 248, 0.08) !important;
        border-radius: 10px !important;
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        padding: 12px 28px !important;
        margin: 0 6px !important;
        border: 1px solid rgba(56, 189, 248, 0.15) !important;
        transition: all 0.3s ease !important;
        font-size: 14px !important;
    }
    
    button[role="tab"]:hover {
        background: rgba(56, 189, 248, 0.2) !important;
        border-color: rgba(56, 189, 248, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    button[aria-selected="true"] {
        background: linear-gradient(135deg, #38bdf8, #06b6d4) !important;
        color: white !important;
        box-shadow: 0 8px 24px rgba(56, 189, 248, 0.3) !important;
        border-color: rgba(56, 189, 248, 0.6) !important;
    }
    
    /* ========================
       CARDS PREMIUM
       ======================== */
    .card-premium {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 16px;
        padding: 28px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(56, 189, 248, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }
    
    /* ========================
       INPUTS PREMIUM
       ======================== */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        padding: 12px 16px !important;
        font-family: 'Poppins', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: rgba(56, 189, 248, 0.6) !important;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.25) !important;
        background: rgba(15, 23, 42, 0.95) !important;
    }
    
    /* ========================
       ALERT BOXES PREMIUM
       ======================== */
    .stAlert {
        border-radius: 12px !important;
        padding: 18px !important;
        border-left: 4px solid !important;
        backdrop-filter: blur(10px);
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), transparent) !important;
        border-left-color: #22c55e !important;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), transparent) !important;
        border-left-color: #ef4444 !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), transparent) !important;
        border-left-color: #f59e0b !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.1), transparent) !important;
        border-left-color: #38bdf8 !important;
    }
    
    /* ========================
       DATAFRAME PREMIUM
       ======================== */
    div[data-testid="stDataFrame"] {
        background: rgba(15, 23, 42, 0.5) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(56, 189, 248, 0.15) !important;
        overflow: hidden !important;
    }
    
    /* ========================
       HEADINGS
       ======================== */
    h1 {
        color: #e2e8f0 !important;
        font-weight: 800 !important;
        font-size: 40px !important;
        letter-spacing: -1px !important;
        margin-bottom: 20px !important;
    }
    
    h2 {
        color: #e2e8f0 !important;
        font-weight: 700 !important;
        font-size: 32px !important;
        margin-top: 35px !important;
        margin-bottom: 20px !important;
        letter-spacing: -0.5px !important;
    }
    
    h3 {
        color: #cbd5e1 !important;
        font-weight: 700 !important;
        font-size: 22px !important;
        margin-top: 25px !important;
        margin-bottom: 15px !important;
    }
    
    /* ========================
       SCROLLBAR
       ======================== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(56, 189, 248, 0.3);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(56, 189, 248, 0.5);
    }
    
    /* ========================
       DIVIDER
       ======================== */
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.2), transparent);
        margin: 35px 0;
    }
    
    /* ========================
       FOOTER
       ======================== */
    .footer-premium {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.5), transparent);
        border-top: 1px solid rgba(56, 189, 248, 0.1);
        padding: 40px;
        text-align: center;
        color: #64748b;
        margin-top: 60px;
        border-radius: 16px;
    }
    
    .footer-premium p {
        margin: 8px 0;
        font-weight: 500;
    }
    
    /* ========================
       CUSTOM BADGES
       ======================== */
    .badge-success {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(34, 197, 94, 0.1));
        color: #22c55e;
        padding: 8px 16px;
        border-radius: 8px;
        border: 1px solid rgba(34, 197, 94, 0.3);
        display: inline-block;
        font-weight: 600;
        font-size: 13px;
        margin: 5px 0;
    }
    
    .badge-warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(245, 158, 11, 0.1));
        color: #f59e0b;
        padding: 8px 16px;
        border-radius: 8px;
        border: 1px solid rgba(245, 158, 11, 0.3);
        display: inline-block;
        font-weight: 600;
        font-size: 13px;
        margin: 5px 0;
    }
    
    .badge-error {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(239, 68, 68, 0.1));
        color: #ef4444;
        padding: 8px 16px;
        border-radius: 8px;
        border: 1px solid rgba(239, 68, 68, 0.3);
        display: inline-block;
        font-weight: 600;
        font-size: 13px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)


# ========================
# DATABASE MANAGER
# ========================
class DatabaseManager:
    """Gestionnaire de base de données SQLite"""
    
    def __init__(self):
        self.db_path = "/tmp/specsense.db"
        self.init_database()
    
    def init_database(self):
        """Initialiser la base de données"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mesures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date_heure TIMESTAMP NOT NULL,
                    reference_piece TEXT NOT NULL,
                    operateur TEXT NOT NULL,
                    essai INTEGER NOT NULL,
                    valeur REAL NOT NULL,
                    lsl REAL NOT NULL,
                    usl REAL NOT NULL,
                    machine TEXT NOT NULL,
                    type_defaut TEXT DEFAULT 'OK',
                    severite INTEGER DEFAULT 1,
                    occurrence INTEGER DEFAULT 1,
                    detection INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date_heure, reference_piece, essai)
                )
            """)
            
            conn.commit()
            conn.close()
        except Exception as e:
            st.error(f"Erreur base de donnees: {e}")
    
    def ajouter_mesures(self, mesures: list) -> bool:
        """Ajouter des mesures"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for mesure in mesures:
                cursor.execute("""
                    INSERT INTO mesures 
                    (date_heure, reference_piece, operateur, essai, valeur, 
                     lsl, usl, machine, type_defaut, severite, occurrence, detection)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    mesure['date_heure'],
                    mesure['reference_piece'],
                    mesure['operateur'],
                    mesure['essai'],
                    mesure['valeur'],
                    mesure['lsl'],
                    mesure['usl'],
                    mesure['machine'],
                    mesure['type_defaut'],
                    mesure['severite'],
                    mesure['occurrence'],
                    mesure['detection']
                ))
            
            conn.commit()
            conn.close()
            return True
        except:
            return False
    
    def obtenir_toutes_mesures(self) -> pd.DataFrame:
        """Recuperer toutes les mesures"""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query("SELECT * FROM mesures ORDER BY date_heure DESC", conn)
            conn.close()
            
            if not df.empty:
                df['date_heure'] = pd.to_datetime(df['date_heure'])
                df['valeur'] = df['valeur'].astype(float)
                df['usl'] = df['usl'].astype(float)
                df['lsl'] = df['lsl'].astype(float)
            
            return df
        except:
            return pd.DataFrame()


# ========================
# QUALITY METRICS
# ========================
def calculer_metriques(df: pd.DataFrame) -> dict:
    """Calculer les metriques de qualite"""
    if df.empty:
        return {
            "total": 0, "conforme": 0, "non_conforme": 0,
            "moyenne": 0.0, "ecart_type": 0.0, "usl": 0.0, "lsl": 0.0,
            "cp": 0.0, "cpk": 0.0, "taux_conformite": 0.0, "ppm_defaut": 0,
        }
    
    valeurs = df["valeur"].dropna()
    moyenne = float(valeurs.mean())
    ecart_type = float(valeurs.std()) if len(valeurs) > 1 else 0.0
    usl = float(df["usl"].iloc[0])
    lsl = float(df["lsl"].iloc[0])
    
    tolerance = usl - lsl
    
    if ecart_type > 0:
        cp = tolerance / (6 * ecart_type)
        cpk = min((usl - moyenne) / (3 * ecart_type), (moyenne - lsl) / (3 * ecart_type))
        cpk = max(cpk, 0.0)
    else:
        cp = cpk = 0.0
    
    conforme = len(df[(df['valeur'] <= usl) & (df['valeur'] >= lsl)])
    non_conforme = len(df) - conforme
    taux_conformite = (conforme / len(df) * 100) if len(df) > 0 else 0.0
    ppm_defaut = int((non_conforme / len(df) * 1000000)) if len(df) > 0 else 0
    
    return {
        "total": len(df), "conforme": conforme, "non_conforme": non_conforme,
        "moyenne": moyenne, "ecart_type": ecart_type, "usl": usl, "lsl": lsl,
        "cp": cp, "cpk": cpk, "taux_conformite": taux_conformite, "ppm_defaut": ppm_defaut,
    }


def evaluer_capabilite(cpk: float) -> tuple:
    """Evaluer la capabilite du processus"""
    if cpk >= 1.67:
        return ("EXCELLENT", "#22c55e")
    elif cpk >= 1.33:
        return ("CAPABLE", "#22c55e")
    elif cpk >= 1.0:
        return ("CRITIQUE", "#f59e0b")
    else:
        return ("INCAPABLE", "#ef4444")


# ========================
# MAIN APP
# ========================
@st.cache_resource
def obtenir_db():
    return DatabaseManager()


def main():
    inject_css_premium()
    
    db = obtenir_db()
    df = db.obtenir_toutes_mesures()
    metriques = calculer_metriques(df)
    
    # HEADER
    st.markdown("""
    <div class="header-premium">
        <h1>SpecSense AI</h1>
        <p>Plateforme Intelligente de Gestion de la Qualite Industrielle</p>
    </div>
    """, unsafe_allow_html=True)
    
    # SIDEBAR
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-title">
            <h2>MENU PRINCIPAL</h2>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.radio(
            "Navigation",
            ["Tableau de Bord", "Saisie Mesures", "Analyses SPC", "Capabilite", "Pareto"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### INDICATEURS CLES")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Mesures", metriques['total'])
        with col2:
            st.metric("Conformite", f"{metriques['taux_conformite']:.1f}%")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cpk", f"{metriques['cpk']:.2f}")
        with col2:
            st.metric("PPM Defauts", f"{metriques['ppm_defaut']:,}")
    
    # PAGES
    if page == "Tableau de Bord":
        st.markdown("## Tableau de Bord")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Conformite", f"{metriques['taux_conformite']:.1f}%")
        col2.metric("Conforme", metriques['conforme'])
        col3.metric("Non-Conforme", metriques['non_conforme'])
        col4.metric("Cpk", f"{metriques['cpk']:.2f}")
        col5.metric("PPM", f"{metriques['ppm_defaut']:,}")
        
        st.markdown("---")
        
        etat, couleur = evaluer_capabilite(metriques['cpk'])
        
        if metriques['cpk'] >= 1.33:
            st.success("Processus CAPABLE et maitrise")
        elif metriques['cpk'] >= 1.0:
            st.warning("Processus CRITIQUE - Amelioration requise")
        else:
            st.error("Processus INCAPABLE - Action IMMEDIATE requise")
        
        st.markdown("---")
        
        if not df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Evolution des Mesures")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(df))), y=df['valeur'],
                    mode='lines+markers', name='Mesures',
                    line=dict(color='#38bdf8', width=2),
                    marker=dict(size=6)
                ))
                fig.add_hline(y=metriques['moyenne'], line_dash='dash', 
                             annotation_text='Moyenne', line_color='#10b981')
                fig.add_hline(y=metriques['usl'], line_dash='dot', 
                             annotation_text='USL', line_color='#ef4444')
                fig.add_hline(y=metriques['lsl'], line_dash='dot', 
                             annotation_text='LSL', line_color='#ef4444')
                fig.update_layout(template='plotly_dark', height=400, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Distribution des Valeurs")
                fig = px.histogram(
                    df, x='valeur', nbins=40,
                    template='plotly_dark',
                    color_discrete_sequence=['#38bdf8']
                )
                fig.add_vline(x=metriques['usl'], line_color='red', annotation_text='USL')
                fig.add_vline(x=metriques['lsl'], line_color='red', annotation_text='LSL')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Saisie Mesures":
        st.markdown("## Saisie de Nouvelles Mesures")
        
        with st.form("formulaire_principal", clear_on_submit=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### Information Piece")
                reference = st.text_input("Reference Piece", placeholder="Ex: P001")
                machine = st.selectbox("Machine", ["M1", "M2", "M3", "M4", "M5"])
            
            with col2:
                st.markdown("### Information Operateur")
                operateur = st.selectbox("Operateur", 
                                        ["Ahmed", "Mohamed", "Ali", "Fatima", "Hassan"])
                equipe = st.selectbox("Equipe", ["Matin", "Apres-midi", "Nuit"])
            
            with col3:
                st.markdown("### Limites Acceptables")
                usl = st.number_input("USL", value=12.5000, format="%.4f")
                lsl = st.number_input("LSL", value=11.5000, format="%.4f")
            
            st.markdown("---")
            st.markdown("### Trois Mesures Obligatoires")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                mesure1 = st.number_input("Mesure 1", format="%.4f", key="m1")
            with col2:
                mesure2 = st.number_input("Mesure 2", format="%.4f", key="m2")
            with col3:
                mesure3 = st.number_input("Mesure 3", format="%.4f", key="m3")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                type_defaut = st.selectbox("Type de Defaut", 
                                          ["OK", "Diametre", "Rugosite", "Rayure", "Autre"])
            with col2:
                remarques = st.text_input("Remarques (optionnel)")
            
            col1, col2 = st.columns([1, 4])
            with col1:
                submit = st.form_submit_button("ENREGISTRER", use_container_width=True, type="primary")
            
        if submit:
            if not reference:
                st.error("Erreur: Reference obligatoire")
                return
            
            if usl <= lsl:
                st.error("Erreur: USL doit etre superieur a LSL")
                return
            
            if mesure1 == 0 and mesure2 == 0 and mesure3 == 0:
                st.error("Erreur: Au moins une mesure requise")
                return
            
            mesures = []
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            for essai, valeur in enumerate([mesure1, mesure2, mesure3], 1):
                if valeur != 0:
                    mesures.append({
                        'date_heure': now, 'reference_piece': reference,
                        'operateur': operateur, 'essai': essai, 'valeur': valeur,
                        'lsl': lsl, 'usl': usl, 'machine': machine,
                        'type_defaut': type_defaut, 'severite': 3 if type_defaut != "OK" else 1,
                        'occurrence': 1, 'detection': 1
                    })
            
            if db.ajouter_mesures(mesures):
                st.success(f"Succes: {len(mesures)} mesure(s) enregistree(s)!")
                st.balloons()
                st.rerun()
            else:
                st.error("Erreur lors de l'enregistrement")
    
    elif page == "Analyses SPC":
        st.markdown("## Analyses SPC (Controle Statistique des Processus)")
        
        if df.empty:
            st.warning("Pas de donnees disponibles")
        else:
            mean = metriques['moyenne']
            std = metriques['ecart_type']
            ucl = mean + 3 * std
            lcl = mean - 3 * std
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("CL (Centre)", f"{mean:.4f}")
            col2.metric("UCL (Limite Sup)", f"{ucl:.4f}")
            col3.metric("LCL (Limite Inf)", f"{lcl:.4f}")
            col4.metric("Ecart-Type", f"{std:.4f}")
            
            st.markdown("---")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(df))), y=df['valeur'],
                mode='lines+markers', name='Valeurs',
                line=dict(color='#38bdf8', width=2),
                marker=dict(size=5)
            ))
            fig.add_hline(y=mean, line_dash='dash', name='CL (Centre)')
            fig.add_hline(y=ucl, line_dash='dot', line_color='#ef4444', name='UCL')
            fig.add_hline(y=lcl, line_dash='dot', line_color='#ef4444', name='LCL')
            fig.update_layout(template='plotly_dark', height=450, title="Carte de Controle")
            st.plotly_chart(fig, use_container_width=True)
            
            hors_limites = len(df[(df['valeur'] > ucl) | (df['valeur'] < lcl)])
            if hors_limites > 0:
                st.error(f"Alerte: {hors_limites} point(s) hors controle!")
            else:
                st.success("Tous les points sont sous controle")
    
    elif page == "Capabilite":
        st.markdown("## Analyse de Capabilite")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Cp", f"{metriques['cp']:.2f}")
        col2.metric("Cpk", f"{metriques['cpk']:.2f}")
        col3.metric("USL", f"{metriques['usl']:.4f}")
        col4.metric("LSL", f"{metriques['lsl']:.4f}")
        
        st.markdown("---")
        
        if not df.empty:
            fig = px.histogram(
                df, x='valeur', nbins=40,
                template='plotly_dark',
                color_discrete_sequence=['#38bdf8'],
                title="Distribution des Valeurs"
            )
            fig.add_vline(x=metriques['usl'], line_color='red')
            fig.add_vline(x=metriques['lsl'], line_color='red')
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Pareto":
        st.markdown("## Analyse Pareto")
        
        if df.empty:
            st.warning("Pas de donnees")
        else:
            defauts = df[df['type_defaut'] != 'OK']
            
            if defauts.empty:
                st.success("Aucun defaut detecte!")
            else:
                pareto = defauts['type_defaut'].value_counts().reset_index()
                pareto.columns = ['Type', 'Nombre']
                pareto['Cumul %'] = (pareto['Nombre'].cumsum() / pareto['Nombre'].sum() * 100)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=pareto['Type'], y=pareto['Nombre'],
                    name='Nombre de Defauts',
                    marker_color='#38bdf8'
                ))
                fig.add_trace(go.Scatter(
                    x=pareto['Type'], y=pareto['Cumul %'], 
                    yaxis='y2', mode='lines+markers',
                    name='Cumul %',
                    line=dict(color='#ef4444', width=2)
                ))
                fig.update_layout(
                    yaxis2=dict(side='right', range=[0, 110]),
                    template='plotly_dark',
                    height=450,
                    title="Diagramme de Pareto"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                st.dataframe(pareto, use_container_width=True, hide_index=True)
    
    # FOOTER
    st.markdown("""
    <div class="footer-premium">
        <p><strong>SpecSense AI v2.0</strong> - Plateforme Premium de Qualite</p>
        <p>Production Ready | 2024</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
