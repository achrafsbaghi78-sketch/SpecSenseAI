"""
SpecSense AI - Plateforme Premium de Gestion de la Qualité
Version Française - Production Ready - Clean Code
"""
import os
import sqlite3
from datetime import datetime
from typing import Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ========================
# PAGE CONFIG
# ========================
st.set_page_config(
    page_title="SpecSense AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========================
# CONSTANTS
# ========================
APP_NAME = "SpecSense AI"
APP_VERSION = "V2.0"
DB_PATH = "/tmp/specsense.db"

MENU_ITEMS = [
    "Tableau de Bord",
    "Saisie Mesures",
    "Analyses SPC",
    "Analyse MSA",
    "Capabilite",
    "Pareto",
    "AMDEC",
]

REQUIRED_COLS = [
    "date_heure",
    "reference_piece",
    "operateur",
    "essai",
    "valeur",
    "usl",
    "lsl",
    "machine",
    "type_defaut",
    "severite",
    "occurrence",
    "detection",
]

# ========================
# CSS PREMIUM
# ========================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;900&display=swap');
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html, body, .stApp {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1229 100%) !important;
        color: #e2e8f0;
    }
    
    .stApp {
        background-attachment: fixed;
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1729 0%, #1a1f3a 100%) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.1);
    }
    
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(15,23,42,0.8), rgba(30,41,59,0.6));
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(56, 189, 248, 0.08);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-4px);
    }
    
    div[data-testid="stMetricLabel"] p {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-size: 32px !important;
        font-weight: 900 !important;
        margin-top: 8px !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #38bdf8 0%, #06b6d4 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 32px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 8px 24px rgba(56, 189, 248, 0.3) !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
    }
    
    .stButton > button:hover {
        box-shadow: 0 12px 32px rgba(56, 189, 248, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    
    div[role="tablist"] {
        background: linear-gradient(90deg, rgba(56, 189, 248, 0.05), transparent);
        padding: 15px;
        border-radius: 12px;
        border-bottom: 2px solid rgba(56, 189, 248, 0.2);
    }
    
    button[role="tab"] {
        background: rgba(56, 189, 248, 0.1) !important;
        border-radius: 10px !important;
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        border: 1px solid rgba(56, 189, 248, 0.15) !important;
    }
    
    button[aria-selected="true"] {
        background: linear-gradient(135deg, #38bdf8, #06b6d4) !important;
        color: white !important;
        box-shadow: 0 8px 24px rgba(56, 189, 248, 0.3) !important;
    }
    
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        padding: 12px 16px !important;
    }
    
    h1, h2, h3 {
        color: #e2e8f0 !important;
        font-weight: 700 !important;
    }
    
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.2), transparent);
        margin: 30px 0;
    }
    </style>
    """, unsafe_allow_html=True)


# ========================
# DATABASE MANAGER
# ========================
class DatabaseManager:
    """Gestionnaire de base de donnees SQLite"""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialiser la base de donnees"""
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
            st.error(f"Erreur BD: {e}")
    
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
        except sqlite3.IntegrityError:
            st.error("Erreur: Ces donnees existent deja")
            return False
        except Exception as e:
            st.error(f"Erreur sauvegarde: {e}")
            return False
    
    def obtenir_toutes_mesures(self) -> pd.DataFrame:
        """Recuperer toutes les mesures"""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(
                "SELECT * FROM mesures ORDER BY date_heure DESC",
                conn
            )
            conn.close()
            
            if not df.empty:
                df['date_heure'] = pd.to_datetime(df['date_heure'])
                df['valeur'] = df['valeur'].astype(float)
                df['usl'] = df['usl'].astype(float)
                df['lsl'] = df['lsl'].astype(float)
            
            return df
        except Exception as e:
            st.error(f"Erreur lecture: {e}")
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
    """Evaluer la capabilite"""
    if cpk >= 1.67:
        return ("EXCELLENT", "#22c55e")
    elif cpk >= 1.33:
        return ("CAPABLE", "#22c55e")
    elif cpk >= 1.0:
        return ("CRITIQUE", "#f59e0b")
    else:
        return ("INCAPABLE", "#ef4444")


# ========================
# PAGES
# ========================
@st.cache_resource
def obtenir_db():
    return DatabaseManager()


def page_tableau_bord(df: pd.DataFrame, metriques: dict):
    """Tableau de bord"""
    st.markdown("## Tableau de Bord")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Conformite", f"{metriques['taux_conformite']:.1f}%")
    col2.metric("Conforme", metriques['conforme'])
    col3.metric("Non-Conforme", metriques['non_conforme'])
    col4.metric("Cpk", f"{metriques['cpk']:.2f}")
    col5.metric("PPM", f"{metriques['ppm_defaut']:,}")
    
    st.markdown("---")
    
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
            fig.update_layout(template='plotly_dark', height=400)
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


def page_saisie_mesures(db: DatabaseManager):
    """Saisie de mesures"""
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
        
        col1, col2, col3 = st.columns([1, 1, 2])
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


def page_spc(df: pd.DataFrame, metriques: dict):
    """Analyses SPC"""
    st.markdown("## Analyses SPC (Controle Statistique)")
    
    if df.empty:
        st.warning("Pas de donnees disponibles")
        return
    
    mean = metriques['moyenne']
    std = metriques['ecart_type']
    ucl = mean + 3 * std
    lcl = mean - 3 * std
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CL (Centre)", f"{mean:.4f}")
    col2.metric("UCL (Sup)", f"{ucl:.4f}")
    col3.metric("LCL (Inf)", f"{lcl:.4f}")
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


def page_msa(df: pd.DataFrame, metriques: dict):
    """Analyse MSA"""
    st.markdown("## Analyse Systeme de Mesure (MSA)")
    
    msa_data = df[df['reference_piece'].str.contains('MSA', case=False, na=False)]
    
    if msa_data.empty:
        st.info("Aucune donnee MSA. Ajoutez des pieces avec 'MSA' dans la reference.")
        return
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Mesures MSA", len(msa_data))
    col2.metric("Operateurs", msa_data['operateur'].nunique())
    col3.metric("Pieces", msa_data['reference_piece'].nunique())
    
    st.markdown("---")
    
    mean_msa = float(msa_data["valeur"].mean())
    std_msa = float(msa_data["valeur"].std()) if len(msa_data) > 1 else 0.0
    ref = (metriques['usl'] + metriques['lsl']) / 2
    tolerance = metriques['usl'] - metriques['lsl']
    cg = (0.2 * tolerance) / (6 * std_msa) if std_msa > 0 else 0
    cgk = (0.1 * tolerance - abs(mean_msa - ref)) / (3 * std_msa) if std_msa > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Moyenne MSA", f"{mean_msa:.4f}")
    col2.metric("Ecart-Type", f"{std_msa:.4f}")
    col3.metric("Cg", f"{cg:.2f}")
    col4.metric("Cgk", f"{cgk:.2f}")
    
    if cgk < 1:
        st.error("Systeme de mesure NON acceptable (Cgk < 1)")
    elif cgk < 1.33:
        st.warning("Systeme limite (amelioration recommandee)")
    else:
        st.success("Systeme de mesure acceptable")


def page_capabilite(df: pd.DataFrame, metriques: dict):
    """Analyse de Capabilite"""
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


def page_pareto(df: pd.DataFrame):
    """Analyse Pareto"""
    st.markdown("## Analyse Pareto")
    
    if df.empty:
        st.warning("Pas de donnees")
        return
    
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


def page_amdec(df: pd.DataFrame):
    """Analyse AMDEC"""
    st.markdown("## Analyse des Modes de Defaillance (AMDEC)")
    
    df_amdec = df.copy()
    df_amdec['RPN'] = df_amdec['severite'] * df_amdec['occurrence'] * df_amdec['detection']
    df_amdec = df_amdec.sort_values('RPN', ascending=False)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("RPN Maximum", int(df_amdec['RPN'].max()))
    col2.metric("RPN Moyen", f"{df_amdec['RPN'].mean():.1f}")
    col3.metric("Risques Critiques", len(df_amdec[df_amdec['RPN'] >= 150]))
    
    st.markdown("---")
    
    affichage = df_amdec[['reference_piece', 'type_defaut', 'severite', 'occurrence', 'detection', 'RPN']].head(20)
    affichage.columns = ['Reference', 'Type Defaut', 'Severite', 'Occurrence', 'Detection', 'RPN']
    
    st.dataframe(affichage, use_container_width=True, hide_index=True)


# ========================
# MAIN
# ========================
def main():
    inject_css()
    
    db = obtenir_db()
    df = db.obtenir_toutes_mesures()
    metriques = calculer_metriques(df)
    
    # HEADER
    st.markdown("""
    <div style="
        padding: 40px;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 20px;
        border: 1px solid rgba(59, 130, 246, 0.2);
        margin-bottom: 30px;
    ">
        <h1 style="
            background: linear-gradient(135deg, #38bdf8 0%, #06b6d4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 48px;
            font-weight: 900;
            margin: 0;
        ">SpecSense AI</h1>
        <p style="color: #94a3b8; font-size: 16px; margin: 10px 0 0 0;">
            Plateforme Intelligente de Gestion de la Qualite Industrielle
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # SIDEBAR
    with st.sidebar:
        st.markdown("### MENU PRINCIPAL")
        
        page = st.radio(
            "Navigation",
            MENU_ITEMS,
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
        page_tableau_bord(df, metriques)
    
    elif page == "Saisie Mesures":
        page_saisie_mesures(db)
    
    elif page == "Analyses SPC":
        page_spc(df, metriques)
    
    elif page == "Analyse MSA":
        page_msa(df, metriques)
    
    elif page == "Capabilite":
        page_capabilite(df, metriques)
    
    elif page == "Pareto":
        page_pareto(df)
    
    elif page == "AMDEC":
        page_amdec(df)
    
    # FOOTER
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #64748b;">
        <p><strong>SpecSense AI v2.0</strong> | Plateforme de Gestion de la Qualite</p>
        <p>Production Ready | 2024</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
