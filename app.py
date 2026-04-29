import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="SpecSense AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# STYLE PRO
# =========================
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
    background: linear-gradient(180deg, #020617, #0b1220);
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

.hero {
    display: flex;
    align-items: center;
    gap: 22px;
    margin-bottom: 25px;
}

.hero img {
    width: 150px;
}

.hero h1 {
    font-size: 44px;
    font-weight: 900;
    margin: 0;
    color: white;
}

.hero p {
    margin: 8px 0 0 0;
    color: #94a3b8;
    font-size: 18px;
}

@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero {
        flex-direction: column;
        text-align: center;
    }

    .hero img {
        width: 180px;
    }

    .hero h1 {
        font-size: 34px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 34px !important;
    }
}
</style>
""", unsafe_allow_html=True)

# =========================
# GOOGLE SHEET
# =========================
G_SHEET_URL = "https://docs.google.com/spreadsheets/d/1Xy4tgkGs1OXOTh-OMAsR7YsfkUPxttF7qalhDdhHa90/export?format=csv&gid=0"

@st.cache_data(ttl=60)
def load_data():
    df = pd.read_csv(G_SHEET_URL)

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
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid_rows = df[df[numeric_cols].isna().any(axis=1)]

    if len(invalid_rows) > 0:
        st.error("❌ Erreur data : certaines valeurs numériques sont invalides.")
        st.dataframe(invalid_rows, use_container_width=True)
        st.stop()

    return df

try:
    df = load_data()
except Exception as e:
    st.error("🚨 Impossible de lire Google Sheet.")
    st.write(e)
    st.stop()

if df.empty:
    st.error("❌ Aucune donnée disponible.")
    st.stop()

# =========================
# CALCULS
# =========================
msa_data = df[df["Part_ID"].astype(str).str.contains("MSA", na=False)]
spc_data = df[df["Part_ID"].astype(str).str.contains("SPC", na=False)]

if len(spc_data) == 0:
    spc_data = df.copy()

total = len(df)
msa_count = len(msa_data)
spc_count = len(spc_data)

mean_val = df["Measurement"].mean()
std_val = df["Measurement"].std()
usl = df["USL"].iloc[0]
lsl = df["LSL"].iloc[0]

if std_val > 0:
    cp = (usl - lsl) / (6 * std_val)
    cpk = min(
        (usl - mean_val) / (3 * std_val),
        (mean_val - lsl) / (3 * std_val)
    )
else:
    cp = 0
    cpk = 0

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.image("logo.png", width=160)

    st.markdown("## 📊 Navigation")

    page = st.radio(
        "",
        [
            "📏 MSA",
            "📉 SPC",
            "🎯 Capabilité",
            "📊 Pareto",
            "⚠️ AMDEC",
            "🤖 IA"
        ]
    )

    st.markdown("---")
    st.metric("📦 Total mesures", total)
    st.metric("📏 Points MSA", msa_count)
    st.metric("📈 Points SPC", spc_count)
    st.markdown("---")
    st.caption(f"🕐 Dernière MAJ : {datetime.now().strftime('%H:%M:%S')}")

page_clean = page.split(" ", 1)[1]

# =========================
# HEADER
# =========================
st.markdown("""
<div class="hero">
    <img src="logo.png">
    <div>
        <h1>SpecSense AI</h1>
        <p>Suite Qualité 4.0 • Tableau de bord qualité intelligent</p>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# KPI
# =========================
k1, k2, k3, k4 = st.columns(4)

k1.metric("Moyenne", f"{mean_val:.4f}")
k2.metric("Écart-type", f"{std_val:.6f}")
k3.metric("Cp", f"{cp:.2f}")
k4.metric("Cpk", f"{cpk:.2f}", "Capable ✅" if cpk >= 1.33 else "Non capable ⚠️")

if cpk < 1:
    st.error("🚨 Statut global : Processus non capable")
elif cpk < 1.33:
    st.warning("⚠️ Statut global : Amélioration nécessaire")
else:
    st.success("✅ Statut global : Processus capable")

st.markdown("---")

# =========================
# MSA
# =========================
if page_clean == "MSA":
    st.subheader("📏 Analyse MSA Type 1")

    if len(msa_data) > 0:
        mean_msa = msa_data["Measurement"].mean()
        std_msa = msa_data["Measurement"].std()

        ref = (usl + lsl) / 2
        tolerance = usl - lsl

        cg = (0.2 * tolerance) / (6 * std_msa) if std_msa > 0 else 0
        cgk = (0.1 * tolerance - abs(mean_msa - ref)) / (3 * std_msa) if std_msa > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Référence", f"{ref:.4f}")
        c2.metric("Tolérance", f"{tolerance:.4f}")
        c3.metric("Cg", f"{cg:.2f}")
        c4.metric("Cgk", f"{cgk:.2f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(msa_data) + 1)),
            y=msa_data["Measurement"],
            mode="lines+markers",
            name="Mesures MSA"
        ))
        fig.add_hline(y=mean_msa, line_dash="dash", annotation_text="Moyenne")
        fig.add_hline(y=ref, line_dash="dot", annotation_text="Référence")
        fig.update_layout(title="Carte MSA", template="plotly_dark", height=430)

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(msa_data, use_container_width=True, hide_index=True)

        st.markdown("### 🧠 Interprétation MSA")
        if cg >= 1.33 and cgk >= 1.33:
            st.success("✅ Le système de mesure est acceptable.")
        else:
            st.error("❌ Le système de mesure n’est pas acceptable.")
    else:
        st.warning("Aucune donnée MSA disponible.")

# =========================
# SPC
# =========================
elif page_clean == "SPC":
    st.subheader("📉 Carte de contrôle SPC")

    mean_spc = spc_data["Measurement"].mean()
    std_spc = spc_data["Measurement"].std()

    ucl = mean_spc + 3 * std_spc
    lcl = mean_spc - 3 * std_spc

    c1, c2, c3 = st.columns(3)
    c1.metric("Ligne centrale", f"{mean_spc:.4f}")
    c2.metric("Limite supérieure", f"{ucl:.4f}")
    c3.metric("Limite inférieure", f"{lcl:.4f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(spc_data) + 1)),
        y=spc_data["Measurement"],
        mode="lines+markers",
        name="SPC"
    ))

    fig.add_hline(y=mean_spc, line_dash="dash", annotation_text="CL")
    fig.add_hline(y=ucl, line_dash="dash", annotation_text="UCL")
    fig.add_hline(y=lcl, line_dash="dash", annotation_text="LCL")
    fig.add_hline(y=usl, line_dash="dot", annotation_text="USL")
    fig.add_hline(y=lsl, line_dash="dot", annotation_text="LSL")

    fig.update_layout(title="Carte de contrôle", template="plotly_dark", height=460)
    st.plotly_chart(fig, use_container_width=True)

    out_spec = spc_data[
        (spc_data["Measurement"] > usl) |
        (spc_data["Measurement"] < lsl)
    ]

    st.markdown("### 🧠 Interprétation SPC")
    if len(out_spec) > 0:
        st.error(f"❌ {len(out_spec)} point(s) hors spécifications.")
        st.dataframe(out_spec, use_container_width=True, hide_index=True)
    else:
        st.success("✅ Le processus est sous contrôle par rapport aux spécifications.")

# =========================
# CAPABILITÉ
# =========================
elif page_clean == "Capabilité":
    st.subheader("🎯 Capabilité du processus")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("USL", f"{usl:.4f}")
    c2.metric("LSL", f"{lsl:.4f}")
    c3.metric("Cp", f"{cp:.2f}")
    c4.metric("Cpk", f"{cpk:.2f}")

    fig = px.histogram(
        df,
        x="Measurement",
        nbins=25,
        title="Histogramme de capabilité",
        template="plotly_dark"
    )

    fig.add_vline(x=usl, line_dash="dash", annotation_text="USL")
    fig.add_vline(x=lsl, line_dash="dash", annotation_text="LSL")
    fig.add_vline(x=mean_val, line_dash="dot", annotation_text="Moyenne")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧠 Interprétation Capabilité")
    if cpk >= 1.33:
        st.success("✅ Le processus est capable de respecter les tolérances client.")
    elif cpk >= 1:
        st.warning("⚠️ Processus limite. Réduire la variation.")
    else:
        st.error("❌ Processus non capable. Actions correctives nécessaires.")

# =========================
# PARETO
# =========================
elif page_clean == "Pareto":
    st.subheader("📊 Analyse Pareto des défauts")

    defects = df[df["Defect_Type"].astype(str).str.upper() != "OK"]

    if len(defects) > 0:
        pareto = defects["Defect_Type"].value_counts().reset_index()
        pareto.columns = ["Type de défaut", "Nombre"]
        pareto["Cumul %"] = pareto["Nombre"].cumsum() / pareto["Nombre"].sum() * 100

        fig = go.Figure()
        fig.add_trace(go.Bar(x=pareto["Type de défaut"], y=pareto["Nombre"], name="Défauts"))
        fig.add_trace(go.Scatter(
            x=pareto["Type de défaut"],
            y=pareto["Cumul %"],
            yaxis="y2",
            mode="lines+markers",
            name="Cumul %"
        ))

        fig.update_layout(
            title="Diagramme Pareto",
            template="plotly_dark",
            height=460,
            yaxis=dict(title="Nombre"),
            yaxis2=dict(title="Cumul %", overlaying="y", side="right", range=[0, 110])
        )

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pareto, use_container_width=True, hide_index=True)

        top_defect = pareto.iloc[0]["Type de défaut"]
        top_count = pareto.iloc[0]["Nombre"]

        st.markdown("### 🧠 Interprétation Pareto")
        st.info(f"Le défaut principal est **{top_defect}** avec **{top_count} occurrence(s)**.")
    else:
        st.success("✅ Aucun défaut détecté.")

# =========================
# AMDEC
# =========================
elif page_clean == "AMDEC":
    st.subheader("⚠️ Analyse AMDEC automatique")

    fmea = df.copy()
    fmea["RPN"] = fmea["Severity"] * fmea["Occurrence"] * fmea["Detection"]

    def get_status(rpn):
        if rpn >= 150:
            return "🔴 Critique"
        elif rpn >= 100:
            return "🟡 Élevé"
        else:
            return "🟢 Moyen"

    def get_action(rpn):
        if rpn >= 150:
            return "Action immédiate requise"
        elif rpn >= 100:
            return "Amélioration nécessaire"
        else:
            return "Risque acceptable"

    fmea["Statut"] = fmea["RPN"].apply(get_status)
    fmea["Action"] = fmea["RPN"].apply(get_action)
    fmea = fmea.sort_values(by="RPN", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("RPN maximum", int(fmea["RPN"].max()))
    c2.metric("RPN moyen", f"{fmea['RPN'].mean():.1f}")
    c3.metric("Risques critiques", len(fmea[fmea["RPN"] >= 150]))

    table_fmea = fmea[[
        "Part_ID", "Defect_Type", "Severity",
        "Occurrence", "Detection", "RPN", "Statut", "Action"
    ]].rename(columns={
        "Part_ID": "Référence pièce",
        "Defect_Type": "Type de défaut",
        "Severity": "Gravité",
        "Detection": "Détection"
    })

    st.dataframe(table_fmea, use_container_width=True, hide_index=True)

    max_rpn = fmea["RPN"].max()

    st.markdown("### 🧠 Interprétation AMDEC")
    if max_rpn >= 150:
        st.error("🔴 Risque critique détecté. Action immédiate obligatoire.")
    elif max_rpn >= 100:
        st.warning("🟡 Risque élevé. Amélioration nécessaire.")
    else:
        st.success("🟢 Niveau de risque acceptable.")

# =========================
# IA
# =========================
elif page_clean == "IA":
    st.subheader("🤖 Assistant Qualité IA")

    st.info("Posez des questions sur SPC, Cpk, MSA, Pareto ou AMDEC.")

    user_question = st.text_area("Votre question qualité")

    if st.button("Analyser"):
        if user_question.strip() == "":
            st.warning("Écrivez une question.")
        else:
            st.success("Module IA prêt. Vous pouvez connecter Hugging Face ici.")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("SpecSense AI V1.0 | Qualité 4.0 | Inspiré IATF 16949")
