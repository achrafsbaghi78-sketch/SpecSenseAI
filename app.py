"""
SpecSense AI - Plateforme Intelligente de Gestion de la Qualité
Version Française - Production Ready
"""
import os
import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sqlite3
from datetime import datetime
from pathlib import Path

# ========================
# CONFIGURATION PAGE
# ========================
st.set_page_config(
    page_title="SpecSense AI - Qualité Intelligente",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "SpecSense AI v2.0 - Plateforme de Gestion de la Qualité Industrielle"
    }
)

# ========================
# CSS PROFESSIONNEL
# ========================
def inject_css():
    st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: #f1f5f9;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
        border-right: 2px solid #3b82f6;
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.5);
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    .sidebar-header h2 {
        color: white;
        font-size: 24px;
        font-weight: 900;
        margin: 0;
    }
    
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(59, 130, 246, 0.4);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.1);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        border-color: rgba(59, 130, 246, 0.8);
        box-shadow: 0 12px 40px rgba(59, 130, 246, 0.2);
        transform: translateY(-2px);
    }
    
    div[data-testid="stMetricLabel"] p {
        color: #cbd5e1 !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="stMetricValue"] {
        color: #3b82f6 !important;
        font-size: 32px !important;
        font-weight: 900 !important;
        margin-top: 8px !important;
    }
    
    .pro-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        margin: 15px 0;
    }
    
    .status-excellent {
        color: #22c55e;
        font-weight: 900;
    }
    
    .status-warning {
        color: #f59e0b;
        font-weight: 900;
    }
    
    .status-critical {
        color: #ef4444;
        font-weight: 900;
    }
    
    .header-title {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 42px;
        font-weight: 900;
        margin-bottom: 10px;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 16px;
        font-weight: 500;
    }
    
    div[role="tablist"] {
        background: linear-gradient(90deg, rgba(59, 130, 246, 0.1), transparent);
        padding: 10px;
        border-radius: 10px;
        border-bottom: 2px solid rgba(59, 130, 246, 0.3);
    }
    
    button[role="tab"] {
        background: rgba(59, 130, 246, 0.2) !important;
        border-radius: 8px !important;
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        padding: 12px 20px !important;
        margin: 0 5px !important;
    }
    
    button[aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4) !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ========================
# DATABASE MANAGER
# ========================
class DatabaseManager:
    """Gestionnaire de base de données SQLite"""
    
    def __init__(self):
        db_path = st.secrets.get("DATABASE_PATH", "/tmp/specsense.db")
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialiser la base de données"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Table des mesures
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
            
            # Table du journal d'audit
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS journal_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    utilisateur TEXT,
                    donnees TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
        except Exception as e:
            st.error(f"❌ Erreur base de données: {e}")
    
    def ajouter_mesures(self, mesures: list) -> bool:
        """Ajouter des mesures"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for mesure in mesures:
                # Validation
                if mesure['usl'] <= mesure['lsl']:
                    st.error("❌ LSL doit être < USL")
                    return False
                
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
                
                self._ajouter_journal("AJOUT_MESURE", mesure['operateur'], str(mesure))
            
            conn.commit()
            conn.close()
            return True
        
        except sqlite3.IntegrityError:
            st.error("❌ Ces données existent déjà")
            return False
        except Exception as e:
            st.error(f"❌ Erreur sauvegarde: {e}")
            return False
    
    def _ajouter_journal(self, action: str, utilisateur: str, donnees: str):
        """Ajouter une entrée au journal"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO journal_audit (action, utilisateur, donnees)
                VALUES (?, ?, ?)
            """, (action, utilisateur, donnees))
            conn.commit()
            conn.close()
        except:
            pass
    
    def obtenir_toutes_mesures(self) -> pd.DataFrame:
        """Récupérer toutes les mesures"""
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
            st.error(f"❌ Erreur lecture: {e}")
            return pd.DataFrame()


# ========================
# INDICATEURS QUALITÉ
# ========================
def calculer_metriques(df: pd.DataFrame) -> dict:
    """Calculer tous les indicateurs de qualité"""
    if df.empty:
        return {
            "total": 0,
            "conforme": 0,
            "non_conforme": 0,
            "moyenne": 0.0,
            "ecart_type": 0.0,
            "usl": 0.0,
            "lsl": 0.0,
            "cp": 0.0,
            "cpk": 0.0,
            "taux_conformite": 0.0,
            "ppm_defaut": 0,
        }
    
    valeurs = df["valeur"].dropna()
    moyenne = float(valeurs.mean())
    ecart_type = float(valeurs.std()) if len(valeurs) > 1 else 0.0
    usl = float(df["usl"].iloc[0])
    lsl = float(df["lsl"].iloc[0])
    
    tolerance = usl - lsl
    
    if ecart_type > 0:
        cp = tolerance / (6 * ecart_type)
        cpk = min(
            (usl - moyenne) / (3 * ecart_type),
            (moyenne - lsl) / (3 * ecart_type)
        )
        cpk = max(cpk, 0.0)
    else:
        cp = 0.0
        cpk = 0.0
    
    conforme = len(df[(df['valeur'] <= usl) & (df['valeur'] >= lsl)])
    non_conforme = len(df) - conforme
    taux_conformite = (conforme / len(df) * 100) if len(df) > 0 else 0.0
    ppm_defaut = int((non_conforme / len(df) * 1000000)) if len(df) > 0 else 0
    
    return {
        "total": len(df),
        "conforme": conforme,
        "non_conforme": non_conforme,
        "moyenne": moyenne,
        "ecart_type": ecart_type,
        "usl": usl,
        "lsl": lsl,
        "cp": cp,
        "cpk": cpk,
        "taux_conformite": taux_conformite,
        "ppm_defaut": ppm_defaut,
    }


def evaluer_capabilite(cpk: float) -> tuple:
    """Évaluer la capabilité du processus"""
    if cpk >= 1.67:
        return ("🌟 EXCELLENT", "#22c55e")
    elif cpk >= 1.33:
        return ("✅ CAPABLE", "#22c55e")
    elif cpk >= 1.0:
        return ("🟡 CRITIQUE", "#f59e0b")
    else:
        return ("❌ INCAPABLE", "#ef4444")


# ========================
# PAGES
# ========================
@st.cache_resource
def obtenir_gestionnaire_db():
    return DatabaseManager()


def render_header():
    """Afficher l'en-tête"""
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            padding: 15px;
            border-radius: 12px;
            text-align: center;
        ">
            <span style="font-size: 40px;">🎯</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div>
            <h1 style="
                background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-size: 40px;
                font-weight: 900;
                margin: 0;
            ">SpecSense AI</h1>
            <p style="
                color: #94a3b8;
                font-size: 16px;
                margin: 5px 0 0 0;
                font-weight: 500;
            ">Plateforme Intelligente de Gestion de la Qualité Industrielle</p>
        </div>
        """, unsafe_allow_html=True)


def render_sidebar(metriques: dict) -> str:
    """Afficher la barre latérale"""
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h2>📋 MENU</h2>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.radio(
            "Navigation",
            [
                "📊 Tableau de Bord",
                "➕ Saisie de Mesures",
                "📈 Analyses SPC",
                "🔍 Capabilité",
                "📉 Pareto",
                "⚙️ MSA",
                "⚠️ AMDEC"
            ],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 📌 INDICATEURS CLÉS")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total", metriques.get('total', 0), "mesures")
        with col2:
            st.metric("Conforme", f"{metriques.get('taux_conformite', 0):.1f}%", "✅")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cpk", f"{metriques.get('cpk', 0):.2f}", "Index")
        with col2:
            st.metric("PPM", f"{metriques.get('ppm_defaut', 0):,}", "Défauts")
        
        st.markdown("---")
        st.markdown("### 🔔 STATUT")
        
        etat, couleur = evaluer_capabilite(metriques.get('cpk', 0))
        st.markdown(f"<p style='color: {couleur}; font-weight: 900;'>{etat}</p>", 
                   unsafe_allow_html=True)
        
        st.markdown("---")
        st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
        st.caption("v2.0 - Production Ready")
    
    return page


def page_tableau_bord(df: pd.DataFrame, metriques: dict):
    """Page Tableau de Bord"""
    st.subheader("📊 Tableau de Bord")
    
    # KPIs Principaux
    st.markdown("### 📈 INDICATEURS PRINCIPAUX")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Taux de Conformité",
            f"{metriques['taux_conformite']:.1f}%",
            f"+{metriques['conforme']}"
        )
    
    with col2:
        st.metric(
            "Pièces Conformes",
            metriques['conforme'],
            "✅"
        )
    
    with col3:
        st.metric(
            "Pièces Non-Conformes",
            metriques['non_conforme'],
            "❌" if metriques['non_conforme'] > 0 else "✅"
        )
    
    with col4:
        st.metric(
            "Cpk",
            f"{metriques['cpk']:.2f}",
            "Indice de Capabilité"
        )
    
    with col5:
        st.metric(
            "PPM",
            f"{metriques['ppm_defaut']:,}",
            "Défauts par Million"
        )
    
    st.markdown("---")
    
    # Statut Global
    st.markdown("### 🎯 STATUT GLOBAL DU PROCESSUS")
    
    etat, couleur = evaluer_capabilite(metriques['cpk'])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if metriques['cpk'] >= 1.33:
            st.success("✅ Le processus est capable et maîtrisé")
        elif metriques['cpk'] >= 1.0:
            st.warning("⚠️ Le processus approche de la limite. Amélioration requise.")
        else:
            st.error("❌ Le processus n'est pas capable. Action immédiate requise!")
    
    with col2:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {couleur}40, {couleur}20);
            border: 2px solid {couleur};
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        ">
            <p style="color: {couleur}; font-weight: 900; font-size: 24px; margin: 0;">
                {etat.split()[1]}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Graphiques
    st.markdown("### 📊 VISUALISATIONS")
    
    if not df.empty:
        tab1, tab2, tab3 = st.tabs(["📈 Évolution", "📊 Distribution", "👥 Opérateurs"])
        
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(df))),
                y=df['valeur'],
                mode='lines+markers',
                name='Mesures',
                line=dict(color='#3b82f6', width=2),
                marker=dict(size=6)
            ))
            fig.add_hline(y=metriques['moyenne'], line_dash='dash', 
                         annotation_text='Moyenne', line_color='#10b981')
            fig.add_hline(y=metriques['usl'], line_dash='dot', 
                         annotation_text='USL', line_color='#ef4444')
            fig.add_hline(y=metriques['lsl'], line_dash='dot', 
                         annotation_text='LSL', line_color='#ef4444')
            fig.update_layout(
                title='Évolution des Mesures',
                template='plotly_dark',
                height=450,
                hovermode='x unified',
                xaxis_title='Numéro de Mesure',
                yaxis_title='Valeur'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = px.histogram(
                df,
                x='valeur',
                nbins=40,
                title='Distribution des Valeurs',
                template='plotly_dark',
                color_discrete_sequence=['#3b82f6']
            )
            fig.add_vline(x=metriques['usl'], line_dash='dash', line_color='red',
                         annotation_text='USL')
            fig.add_vline(x=metriques['lsl'], line_dash='dash', line_color='red',
                         annotation_text='LSL')
            fig.add_vline(x=metriques['moyenne'], line_dash='dot', line_color='green',
                         annotation_text='Moyenne')
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            if 'operateur' in df.columns:
                fig = px.box(
                    df,
                    x='operateur',
                    y='valeur',
                    color='operateur',
                    title='Performance par Opérateur',
                    template='plotly_dark'
                )
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)
    
    # Tableau des dernières mesures
    st.markdown("---")
    st.markdown("### 📋 DERNIÈRES MESURES")
    
    if not df.empty:
        affichage_df = df[['date_heure', 'reference_piece', 'operateur', 'valeur', 'machine']].head(15).copy()
        affichage_df.columns = ['Date/Heure', 'Référence', 'Opérateur', 'Valeur', 'Machine']
        
        # Colorier le statut
        st.dataframe(
            affichage_df,
            use_container_width=True,
            hide_index=True
        )


def page_saisie_mesures(gestionnaire: DatabaseManager):
    """Page Saisie de Mesures"""
    st.subheader("➕ Saisie de Nouvelles Mesures")
    
    st.markdown("""
    <div class="pro-card">
        <p>Enregistrez les mesures de vos pièces en temps réel avec validation automatique.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mode de saisie
    mode = st.radio("Mode de saisie", ["Saisie Manuelle", "Importer Excel"], horizontal=True)
    
    if mode == "Saisie Manuelle":
        with st.form("formulaire_mesures", clear_on_submit=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 📦 PIÈCE")
                type_data = st.selectbox("Type", ["SPC", "MSA"])
                reference = st.text_input("Référence Pièce *").strip()
                machine = st.selectbox("Machine", ["M1", "M2", "M3", "M4", "M5"])
            
            with col2:
                st.markdown("### 👤 OPÉRATEUR")
                operateur = st.selectbox("Opérateur", 
                                        ["Ahmed", "Mohamed", "Ali", "Fatima", "Hassan"])
                equipe = st.selectbox("Équipe", ["Matin", "Après-midi", "Nuit"])
            
            with col3:
                st.markdown("### ⚙️ LIMITES")
                usl = st.number_input("USL (Limite Sup.)", value=12.5000, format="%.4f")
                lsl = st.number_input("LSL (Limite Inf.)", value=11.5000, format="%.4f")
            
            st.markdown("---")
            st.markdown("### 📏 TROIS MESURES OBLIGATOIRES")
            
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
                type_defaut = st.selectbox("Type de Défaut", 
                                          ["OK", "Diamètre", "Rugosité", "Rayure", "Autre"])
            with col2:
                remarques = st.text_input("Remarques")
            
            st.markdown("---")
            
            submit = st.form_submit_button(
                "✅ ENREGISTRER MESURES",
                use_container_width=True,
                type="primary"
            )
        
        if submit:
            # Validation
            if not reference:
                st.error("❌ La référence est obligatoire")
                return
            
            if usl <= lsl:
                st.error("❌ USL doit être > LSL")
                return
            
            if mesure1 == 0 and mesure2 == 0 and mesure3 == 0:
                st.error("❌ Au moins une mesure requise")
                return
            
            mesures = []
            maintenant = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ref_finale = f"{type_data}_{reference}"
            
            for essai, valeur in enumerate([mesure1, mesure2, mesure3], 1):
                if valeur != 0:
                    mesures.append({
                        'date_heure': maintenant,
                        'reference_piece': ref_finale,
                        'operateur': operateur,
                        'essai': essai,
                        'valeur': valeur,
                        'lsl': lsl,
                        'usl': usl,
                        'machine': machine,
                        'type_defaut': type_defaut,
                        'severite': 3 if type_defaut != "OK" else 1,
                        'occurrence': 1,
                        'detection': 1
                    })
            
            if gestionnaire.ajouter_mesures(mesures):
                st.success(f"✅ {len(mesures)} mesure(s) enregistrée(s) avec succès!")
                st.balloons()
                st.rerun()
    
    else:
        st.markdown("### 📥 IMPORTER UN FICHIER EXCEL")
        uploaded = st.file_uploader("Sélectionnez un fichier Excel", type=['xlsx', 'xls'])
        
        if uploaded:
            try:
                df_import = pd.read_excel(uploaded)
                st.dataframe(df_import, use_container_width=True)
                
                if st.button("✅ IMPORTER LES DONNÉES", use_container_width=True, type="primary"):
                    mesures = []
                    for idx, row in df_import.iterrows():
                        mesures.append({
                            'date_heure': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'reference_piece': str(row.get('reference_piece', f'P{idx}')),
                            'operateur': str(row.get('operateur', 'Unknown')),
                            'essai': int(row.get('essai', 1)),
                            'valeur': float(row.get('valeur', 0)),
                            'lsl': float(row.get('lsl', 0)),
                            'usl': float(row.get('usl', 0)),
                            'machine': str(row.get('machine', 'M1')),
                            'type_defaut': str(row.get('type_defaut', 'OK')),
                            'severite': int(row.get('severite', 1)),
                            'occurrence': int(row.get('occurrence', 1)),
                            'detection': int(row.get('detection', 1))
                        })
                    
                    if gestionnaire.ajouter_mesures(mesures):
                        st.success(f"✅ {len(mesures)} enregistrements importés!")
                        st.rerun()
            except Exception as e:
                st.error(f"❌ Erreur importation: {e}")


def page_analyses_spc(df: pd.DataFrame, metriques: dict):
    """Page Analyses SPC"""
    st.subheader("📈 Analyses SPC (Contrôle Statistique)")
    
    if df.empty:
        st.warning("⚠️ Pas de données disponibles")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs(["Carte de Contrôle", "Règles SPC", "Capabilité", "Machines"])
    
    with tab1:
        st.markdown("### 📊 Carte de Contrôle X̄")
        
        moyenne = metriques['moyenne']
        ecart_type = metriques['ecart_type']
        ucl = moyenne + 3 * ecart_type
        lcl = moyenne - 3 * ecart_type
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("CL (Centre)", f"{moyenne:.4f}")
        col2.metric("UCL (Sup.)", f"{ucl:.4f}")
        col3.metric("LCL (Inf.)", f"{lcl:.4f}")
        col4.metric("Écart-Type", f"{ecart_type:.4f}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(df))),
            y=df['valeur'],
            mode='lines+markers',
            name='Valeurs',
            line=dict(color='#3b82f6', width=2)
        ))
        fig.add_hline(y=moyenne, line_dash='dash', name='CL')
        fig.add_hline(y=ucl, line_dash='dot', name='UCL', line_color='#ef4444')
        fig.add_hline(y=lcl, line_dash='dot', name='LCL', line_color='#ef4444')
        fig.update_layout(
            title='Carte de Contrôle',
            template='plotly_dark',
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
        
        hors_limites = df[(df['valeur'] > ucl) | (df['valeur'] < lcl)]
        if not hors_limites.empty:
            st.error(f"⚠️ {len(hors_limites)} point(s) hors contrôle détecté(s)")
        else:
            st.success("✅ Tous les points sont sous contrôle")
    
    with tab2:
        st.markdown("### 🚦 RÈGLES SPC")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hors_limites = len(df[(df['valeur'] > ucl) | (df['valeur'] < lcl)])
            if hors_limites > 0:
                st.error(f"❌ Règle 1: {hors_limites} point(s)")
            else:
                st.success("✅ Règle 1: OK")
        
        with col2:
            st.info("📋 Règle 2: 7 points côté = Tendance")
        
        with col3:
            st.info("📋 Règle 3: Tendance = Alert")
    
    with tab3:
        st.markdown("### 📊 INDICES DE CAPABILITÉ")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Cp", f"{metriques['cp']:.2f}")
        col2.metric("Cpk", f"{metriques['cpk']:.2f}")
        col3.metric("Pp", f"{metriques['cp']:.2f}")
        col4.metric("Ppk", f"{metriques['cpk']:.2f}")
    
    with tab4:
        if 'machine' in df.columns:
            stats_machine = df.groupby('machine')['valeur'].agg(['count', 'mean', 'std']).reset_index()
            st.dataframe(stats_machine, use_container_width=True)
            
            fig = px.box(df, x='machine', y='valeur', color='machine', template='plotly_dark',
                        title='Variation par Machine')
            st.plotly_chart(fig, use_container_width=True)


def page_capabilite(df: pd.DataFrame, metriques: dict):
    """Page Capabilité"""
    st.subheader("🔍 Analyse de Capabilité")
    
    if df.empty:
        st.warning("⚠️ Pas de données")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cp", f"{metriques['cp']:.2f}")
    col2.metric("Cpk", f"{metriques['cpk']:.2f}")
    col3.metric("USL", f"{metriques['usl']:.4f}")
    col4.metric("LSL", f"{metriques['lsl']:.4f}")
    
    st.markdown("---")
    
    etat, couleur = evaluer_capabilite(metriques['cpk'])
    st.markdown(f"<h3 style='color: {couleur};'>{etat}</h3>", unsafe_allow_html=True)
    
    # Histogramme avec limites
    fig = px.histogram(df, x='valeur', nbins=40, template='plotly_dark', title='Distribution')
    fig.add_vline(x=metriques['usl'], line_color='red', annotation_text='USL')
    fig.add_vline(x=metriques['lsl'], line_color='red', annotation_text='LSL')
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)


def page_pareto(df: pd.DataFrame):
    """Page Pareto"""
    st.subheader("📉 Analyse Pareto")
    
    defauts = df[df['type_defaut'] != 'OK']
    
    if defauts.empty:
        st.success("✅ Aucun défaut détecté")
        return
    
    pareto_data = defauts['type_defaut'].value_counts().reset_index()
    pareto_data.columns = ['Type', 'Nombre']
    pareto_data['Cumul %'] = (pareto_data['Nombre'].cumsum() / pareto_data['Nombre'].sum() * 100)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=pareto_data['Type'], y=pareto_data['Nombre'], name='Défauts'))
    fig.add_trace(go.Scatter(x=pareto_data['Type'], y=pareto_data['Cumul %'], yaxis='y2', 
                            mode='lines+markers', name='Cumul %'))
    fig.update_layout(
        yaxis2=dict(side='right', range=[0, 110]),
        template='plotly_dark',
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(pareto_data, use_container_width=True)


def page_msa(df: pd.DataFrame):
    """Page MSA"""
    st.subheader("⚙️ Analyse Système de Mesure (MSA)")
    
    msa_data = df[df['reference_piece'].str.contains('MSA', case=False, na=False)]
    
    if msa_data.empty:
        st.info("ℹ️ Aucune donnée MSA. Ajoutez des pièces avec 'MSA' dans la référence.")
        return
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Mesures MSA", len(msa_data))
    col2.metric("Opérateurs", msa_data['operateur'].nunique())
    col3.metric("Pièces", msa_data['reference_piece'].nunique())
    
    st.markdown("---")
    st.info("📋 Analyse MSA en cours de développement...")


def page_amdec(df: pd.DataFrame):
    """Page AMDEC"""
    st.subheader("⚠️ Analyse des Modes de Défaillance (AMDEC)")
    
    df_amdec = df.copy()
    df_amdec['RPN'] = df_amdec['severite'] * df_amdec['occurrence'] * df_amdec['detection']
    df_amdec = df_amdec.sort_values('RPN', ascending=False)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("RPN Max", int(df_amdec['RPN'].max()))
    col2.metric("RPN Moyen", f"{df_amdec['RPN'].mean():.0f}")
    col3.metric("Risques Critiques", len(df_amdec[df_amdec['RPN'] >= 150]))
    
    st.markdown("---")
    
    affichage = df_amdec[['reference_piece', 'type_defaut', 'severite', 'occurrence', 'detection', 'RPN']].head(20)
    affichage.columns = ['Référence', 'Type Défaut', 'Sévérité', 'Occurrence', 'Détection', 'RPN']
    
    st.dataframe(affichage, use_container_width=True)


def main():
    """Fonction principale"""
    inject_css()
    
    # Initialiser le gestionnaire
    gestionnaire = obtenir_gestionnaire_db()
    
    # Charger les données
    df = gestionnaire.obtenir_toutes_mesures()
    
    # Calculer les métriques
    metriques = calculer_metriques(df)
    
    # Afficher l'interface
    render_header()
    page = render_sidebar(metriques)
    
    # Pages
    if page == "📊 Tableau de Bord":
        page_tableau_bord(df, metriques)
    
    elif page == "➕ Saisie de Mesures":
        page_saisie_mesures(gestionnaire)
    
    elif page == "📈 Analyses SPC":
        page_analyses_spc(df, metriques)
    
    elif page == "🔍 Capabilité":
        page_capabilite(df, metriques)
    
    elif page == "📉 Pareto":
        page_pareto(df)
    
    elif page == "⚙️ MSA":
        page_msa(df)
    
    elif page == "⚠️ AMDEC":
        page_amdec(df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #64748b;">
        <p><strong>🎯 SpecSense AI v2.0</strong> | Plateforme de Gestion de la Qualité</p>
        <p>Production Ready | © 2024 | France</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
