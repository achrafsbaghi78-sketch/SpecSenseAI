import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from huggingface_hub import InferenceClient
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Image


st.set_page_config(
    page_title="SpecSense AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# STYLE
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


# =========================
# AI FUNCTIONS
# =========================
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
                    "content": "Tu es un expert qualité automobile. Réponds en français simple, professionnel et avec des actions concrètes."
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            max_tokens=600,
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Erreur IA : {e}"


def generate_ai_module_analysis(module_name, context):
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

Réponse en français, claire, professionnelle et concrète.
"""
    return ask_hf_ai(prompt)


def show_ai_analysis(module_name, context):
    st.markdown("### 🤖 Interprétation IA & Actions recommandées")

    key = f"ai_{module_name}_{abs(hash(context))}"

    if key not in st.session_state:
        with st.spinner(f"🤖 Analyse IA {module_name}..."):
            st.session_state[key] = generate_ai_module_analysis(module_name, context)

    st.info(st.session_state[key])


# =========================
# DATA
# =========================
G_SHEET_URL = "https://docs.google.com/spreadsheets/d/1Xy4tgkGs1OXOTh-OMAsR7YsfkUPxttF7qalhDdhHa90/export?format=csv&gid=0"


@st.cache_data(ttl=60)
def load_data():
    df = pd.read_csv(G_SHEET_URL)
    df.columns = df.columns.str.strip()

    required_cols = [
        "Date_Time", "Part_ID", "Operator", "Trial",
        "Measurement", "USL", "LSL", "Machine",
        "Defect_Type", "Severity", "Occurrence", "Detection"
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Colonnes manquantes : {missing_cols}")
        st.stop()

    numeric_cols = ["Measurement", "USL", "LSL", "Severity", "Occurrence", "Detection"]

    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(",", ".", regex=False).str.strip()
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
# PDF
# =========================
def generate_pdf_report():
    doc_path = "rapport_qualite_specsense.pdf"
    styles = getSampleStyleSheet()
    story = []

    if os.path.exists("logo.png"):
        story.append(Image("logo.png", width=2*inch, height=1*inch))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Rapport Qualité - SpecSense AI", styles["Title"]))
    story.append(Spacer(1, 20))

    story.append(Paragraph("Indicateurs clés", styles["Heading2"]))
    story.append(Paragraph(f"Nombre total de mesures : {total}", styles["BodyText"]))
    story.append(Paragraph(f"Moyenne : {mean_val:.4f}", styles["BodyText"]))
    story.append(Paragraph(f"Écart-type : {std_val:.6f}", styles["BodyText"]))
    story.append(Paragraph(f"Cp : {cp:.2f}", styles["BodyText"]))
    story.append(Paragraph(f"Cpk : {cpk:.2f}", styles["BodyText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Conclusion", styles["Heading2"]))
    if cpk < 1:
        story.append(Paragraph("Le processus n'est pas conforme aux exigences qualité. Des actions immédiates sont nécessaires.", styles["BodyText"]))
    elif cpk < 1.33:
        story.append(Paragraph("Le processus est limite. Une amélioration est nécessaire.", styles["BodyText"]))
    else:
        story.append(Paragraph("Le processus est globalement maîtrisé.", styles["BodyText"]))

    doc = SimpleDocTemplate(doc_path)
    doc.build(story)

    return doc_path


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=170)
    else:
        st.markdown("## SpecSense AI")

    st.markdown("""
    <div style="font-size:14px;color:#94a3b8;font-weight:800;letter-spacing:1px;margin:18px 0 12px 0;">
        MENU PRINCIPAL
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "",
        [
            "🏠 Tableau de bord",
            "📏 MSA",
            "📉 SPC",
            "🎯 Capabilité",
            "📊 Pareto",
            "⚠️ AMDEC",
            "🤖 IA"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 📌 Indicateurs")
    st.metric("Total mesures", total)
    st.metric("Points MSA", msa_count)
    st.metric("Points SPC", spc_count)
    st.markdown("---")
    st.caption(f"🕐 Dernière MAJ : {datetime.now().strftime('%H:%M:%S')}")

page_clean = page.split(" ", 1)[1]


# =========================
# HEADER
# =========================
h1, h2 = st.columns([1, 5])

with h1:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=140)

with h2:
    st.markdown("""
    <h1 style="margin:0; font-size:44px; font-weight:900;">SpecSense AI</h1>
    <p style="margin:6px 0 0 0; color:#94a3b8; font-size:18px;">
        Plateforme intelligente de qualité industrielle
    </p>
    """, unsafe_allow_html=True)

st.markdown("---")


# =========================
# KPI GLOBAL
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
# TABLEAU DE BORD
# =========================
if page_clean == "Tableau de bord":
    st.subheader("🏠 Vue générale")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 📈 Évolution des mesures")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(df) + 1)),
            y=df["Measurement"],
            mode="lines+markers",
            name="Mesures"
        ))
        fig.add_hline(y=mean_val, line_dash="dash", annotation_text="Moyenne")
        fig.add_hline(y=usl, line_dash="dot", annotation_text="USL")
        fig.add_hline(y=lsl, line_dash="dot", annotation_text="LSL")
        fig.update_layout(template="plotly_dark", height=420)

        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("### 🎯 Distribution")

        fig = px.histogram(
            df,
            x="Measurement",
            nbins=25,
            template="plotly_dark",
            title="Distribution des mesures"
        )
        fig.add_vline(x=usl, line_dash="dash", annotation_text="USL")
        fig.add_vline(x=lsl, line_dash="dash", annotation_text="LSL")
        fig.add_vline(x=mean_val, line_dash="dot", annotation_text="Moyenne")

        st.plotly_chart(fig, use_container_width=True)

    context_dashboard = f"""
Moyenne = {mean_val:.4f}
Écart-type = {std_val:.6f}
Cp = {cp:.2f}
Cpk = {cpk:.2f}
USL = {usl:.4f}
LSL = {lsl:.4f}
Nombre total de mesures = {total}
"""
    show_ai_analysis("Tableau de bord", context_dashboard)


# =========================
# MSA
# =========================
elif page_clean == "MSA":
    st.subheader("📏 Module MSA complet")

    tab_msa1, tab_grr, tab_bias, tab_linearity, tab_stability, tab_attribute = st.tabs([
        "MSA Type 1",
        "Gage R&R",
        "Bias",
        "Linearity",
        "Stability",
        "Attribute MSA"
    ])

    with tab_msa1:
        st.markdown("### 📏 MSA Type 1")

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
            fig.update_layout(title="Carte MSA Type 1", template="plotly_dark", height=430)

            st.plotly_chart(fig, use_container_width=True)

            context_msa = f"""
Référence = {ref:.4f}
Tolérance = {tolerance:.4f}
Moyenne MSA = {mean_msa:.4f}
Écart-type MSA = {std_msa:.6f}
Cg = {cg:.2f}
Cgk = {cgk:.2f}
Nombre de mesures MSA = {len(msa_data)}
"""
            show_ai_analysis("MSA Type 1", context_msa)

        else:
            st.warning("Aucune donnée MSA disponible.")

    with tab_grr:
        st.markdown("### ⚙️ Gage R&R")

        if len(msa_data) > 0:
            df_grr = msa_data.copy()

            var_total = df_grr["Measurement"].var()
            var_operator = df_grr.groupby("Operator")["Measurement"].mean().var()
            var_repeat = df_grr.groupby(["Part_ID", "Operator"])["Measurement"].var().mean()

            var_operator = 0 if pd.isna(var_operator) else var_operator
            var_repeat = 0 if pd.isna(var_repeat) else var_repeat

            var_grr = var_operator + var_repeat
            percent_grr = (var_grr / var_total) * 100 if var_total > 0 else 0

            c1, c2, c3 = st.columns(3)
            c1.metric("Variation totale", f"{var_total:.8f}")
            c2.metric("GRR", f"{var_grr:.8f}")
            c3.metric("%GRR", f"{percent_grr:.2f}%")

            fig = px.box(
                df_grr,
                x="Operator",
                y="Measurement",
                color="Operator",
                title="Variation par opérateur",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

            context_grr = f"""
Variation totale = {var_total:.8f}
Variation opérateur = {var_operator:.8f}
Variation répétabilité = {var_repeat:.8f}
GRR = {var_grr:.8f}
%GRR = {percent_grr:.2f}
"""
            show_ai_analysis("Gage R&R", context_grr)

        else:
            st.warning("Aucune donnée MSA disponible.")

    with tab_bias:
        st.markdown("### 🎯 Bias")

        if len(msa_data) > 0:
            reference = st.number_input("Valeur de référence", value=12.0000, format="%.4f")
            mean_bias = msa_data["Measurement"].mean()
            bias = mean_bias - reference

            c1, c2 = st.columns(2)
            c1.metric("Moyenne mesurée", f"{mean_bias:.6f}")
            c2.metric("Bias", f"{bias:.6f}")

            context_bias = f"""
Référence = {reference:.4f}
Moyenne mesurée = {mean_bias:.6f}
Bias = {bias:.6f}
"""
            show_ai_analysis("Bias", context_bias)

        else:
            st.warning("Aucune donnée MSA disponible.")

    with tab_linearity:
        st.markdown("### 📈 Linearity")

        if len(msa_data) > 0:
            df_lin = msa_data.copy()
            df_lin["Reference"] = df_lin.groupby("Part_ID")["Measurement"].transform("mean")
            df_lin["Bias"] = df_lin["Measurement"] - df_lin["Reference"]

            fig = px.scatter(
                df_lin,
                x="Reference",
                y="Bias",
                color="Operator",
                trendline="ols",
                title="Linearity : Bias vs Référence",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

            bias_var = df_lin.groupby("Part_ID")["Bias"].mean().std()
            st.metric("Variation du Bias", f"{bias_var:.6f}")

            context_linearity = f"""
Variation du Bias = {bias_var:.6f}
Nombre de pièces MSA = {df_lin["Part_ID"].nunique()}
Nombre opérateurs = {df_lin["Operator"].nunique()}
"""
            show_ai_analysis("Linearity", context_linearity)

        else:
            st.warning("Aucune donnée MSA disponible.")

    with tab_stability:
        st.markdown("### ⏳ Stability")

        if len(msa_data) > 0:
            df_stab = msa_data.copy()
            df_stab["Date_Time"] = pd.to_datetime(df_stab["Date_Time"], errors="coerce")
            df_stab = df_stab.dropna(subset=["Date_Time"]).sort_values("Date_Time")

            mean_stab = df_stab["Measurement"].mean()
            std_stab = df_stab["Measurement"].std()
            ucl_stab = mean_stab + 3 * std_stab
            lcl_stab = mean_stab - 3 * std_stab

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_stab["Date_Time"],
                y=df_stab["Measurement"],
                mode="lines+markers",
                name="Mesures"
            ))
            fig.add_hline(y=mean_stab, line_dash="dash", annotation_text="Moyenne")
            fig.add_hline(y=ucl_stab, line_dash="dot", annotation_text="UCL")
            fig.add_hline(y=lcl_stab, line_dash="dot", annotation_text="LCL")
            fig.update_layout(title="Stability dans le temps", template="plotly_dark", height=430)

            st.plotly_chart(fig, use_container_width=True)

            out_stab = df_stab[
                (df_stab["Measurement"] > ucl_stab) |
                (df_stab["Measurement"] < lcl_stab)
            ]

            context_stability = f"""
Moyenne stabilité = {mean_stab:.4f}
Écart-type stabilité = {std_stab:.6f}
UCL = {ucl_stab:.4f}
LCL = {lcl_stab:.4f}
Points instables = {len(out_stab)}
"""
            show_ai_analysis("Stability", context_stability)

        else:
            st.warning("Aucune donnée MSA disponible.")

    with tab_attribute:
        st.markdown("### ✅ Attribute MSA")

        df_attr = df.copy()
        df_attr["Decision"] = df_attr["Defect_Type"].astype(str).str.upper().apply(
            lambda x: "OK" if x == "OK" else "NOK"
        )

        ok_count = len(df_attr[df_attr["Decision"] == "OK"])
        nok_count = len(df_attr[df_attr["Decision"] == "NOK"])
        agreement = (ok_count / len(df_attr)) * 100 if len(df_attr) > 0 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("OK", ok_count)
        c2.metric("NOK", nok_count)
        c3.metric("% OK", f"{agreement:.2f}%")

        fig = px.pie(df_attr, names="Decision", title="Répartition OK / NOK", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        context_attribute = f"""
OK = {ok_count}
NOK = {nok_count}
%OK = {agreement:.2f}
Nombre total = {len(df_attr)}
"""
        show_ai_analysis("Attribute MSA", context_attribute)


# =========================
# SPC
# =========================
elif page_clean == "SPC":
    st.subheader("📉 Module SPC complet")

    mean_spc = spc_data["Measurement"].mean()
    std_spc = spc_data["Measurement"].std()
    ucl = mean_spc + 3 * std_spc
    lcl = mean_spc - 3 * std_spc

    spc_work = spc_data.copy().reset_index(drop=True)
    spc_work["Point"] = range(1, len(spc_work) + 1)
    spc_work["Hors_Controle"] = (
        (spc_work["Measurement"] > ucl) |
        (spc_work["Measurement"] < lcl)
    )

    tab_control, tab_rules, tab_distribution, tab_capability, tab_machine, tab_interpretation = st.tabs([
        "Carte de contrôle",
        "Règles SPC",
        "Distribution",
        "Capabilité",
        "Machine / Opérateur",
        "Interprétation"
    ])

    with tab_control:
        st.markdown("### 📈 Carte de contrôle")

        c1, c2, c3 = st.columns(3)
        c1.metric("CL", f"{mean_spc:.4f}")
        c2.metric("UCL", f"{ucl:.4f}")
        c3.metric("LCL", f"{lcl:.4f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=spc_work["Point"],
            y=spc_work["Measurement"],
            mode="lines+markers",
            name="Mesures"
        ))
        fig.add_hline(y=mean_spc, line_dash="dash", annotation_text="CL")
        fig.add_hline(y=ucl, line_dash="dash", annotation_text="UCL")
        fig.add_hline(y=lcl, line_dash="dash", annotation_text="LCL")
        fig.add_hline(y=usl, line_dash="dot", annotation_text="USL")
        fig.add_hline(y=lsl, line_dash="dot", annotation_text="LSL")
        fig.update_layout(title="Carte de contrôle SPC", template="plotly_dark", height=460)

        st.plotly_chart(fig, use_container_width=True)

    with tab_rules:
        st.markdown("### 🚦 Règles SPC")

        out_control = spc_work[spc_work["Hors_Controle"]]
        rule1 = len(out_control)

        values = spc_work["Measurement"].tolist()
        above = [v > mean_spc for v in values]

        max_run = 1
        current_run = 1

        for i in range(1, len(above)):
            if above[i] == above[i - 1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1

        rule2 = max_run >= 7

        trend_detected = False
        for i in range(len(values) - 5):
            segment = values[i:i + 6]
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

        fig = px.histogram(
            spc_work,
            x="Measurement",
            nbins=25,
            title="Histogramme SPC",
            template="plotly_dark"
        )
        fig.add_vline(x=mean_spc, line_dash="dot", annotation_text="Moyenne")
        fig.add_vline(x=usl, line_dash="dash", annotation_text="USL")
        fig.add_vline(x=lsl, line_dash="dash", annotation_text="LSL")

        st.plotly_chart(fig, use_container_width=True)

    with tab_capability:
        st.markdown("### 🎯 Capabilité SPC")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("USL", f"{usl:.4f}")
        c2.metric("LSL", f"{lsl:.4f}")
        c3.metric("Cp", f"{cp:.2f}")
        c4.metric("Cpk", f"{cpk:.2f}")

    with tab_machine:
        st.markdown("### 🏭 Machine / Opérateur")

        col_m, col_o = st.columns(2)

        with col_m:
            machine_stats = spc_work.groupby("Machine")["Measurement"].agg(["count", "mean", "std"]).reset_index()
            machine_stats.columns = ["Machine", "Nombre", "Moyenne", "Écart-type"]
            st.dataframe(machine_stats, use_container_width=True, hide_index=True)

            fig = px.box(spc_work, x="Machine", y="Measurement", color="Machine", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        with col_o:
            operator_stats = spc_work.groupby("Operator")["Measurement"].agg(["count", "mean", "std"]).reset_index()
            operator_stats.columns = ["Opérateur", "Nombre", "Moyenne", "Écart-type"]
            st.dataframe(operator_stats, use_container_width=True, hide_index=True)

            fig = px.box(spc_work, x="Operator", y="Measurement", color="Operator", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    with tab_interpretation:
        context_spc = f"""
Moyenne SPC = {mean_spc:.4f}
Écart-type SPC = {std_spc:.6f}
UCL = {ucl:.4f}
LCL = {lcl:.4f}
Points hors contrôle = {len(spc_work[spc_work["Hors_Controle"]])}
Cp = {cp:.2f}
Cpk = {cpk:.2f}
"""
        show_ai_analysis("SPC", context_spc)


# =========================
# CAPABILITÉ
# =========================
elif page_clean == "Capabilité":
    st.subheader("🎯 Module Capabilité complet")

    tab_kpi, tab_hist, tab_centering, tab_machine, tab_interpretation = st.tabs([
        "Indices Cp / Cpk",
        "Histogramme",
        "Centrage",
        "Machine / Opérateur",
        "Interprétation"
    ])

    with tab_kpi:
        st.markdown("### 📌 Indices de capabilité")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("LS", f"{usl:.4f}")
        c2.metric("LI", f"{lsl:.4f}")
        c3.metric("Cp", f"{cp:.2f}")
        c4.metric("Cpk", f"{cpk:.2f}")

    with tab_hist:
        st.markdown("### 📊 Histogramme de capabilité")

        fig = px.histogram(
            df,
            x="Measurement",
            nbins=25,
            title="Distribution des mesures",
            template="plotly_dark"
        )
        fig.add_vline(x=usl, line_dash="dash", annotation_text="LS")
        fig.add_vline(x=lsl, line_dash="dash", annotation_text="LI")
        fig.add_vline(x=mean_val, line_dash="dot", annotation_text="Moyenne")

        st.plotly_chart(fig, use_container_width=True)

    with tab_centering:
        st.markdown("### 🎯 Analyse du centrage")

        target = (usl + lsl) / 2
        decentrage = mean_val - target

        c1, c2, c3 = st.columns(3)
        c1.metric("Cible", f"{target:.4f}")
        c2.metric("Moyenne", f"{mean_val:.4f}")
        c3.metric("Décalage", f"{decentrage:.6f}")

    with tab_machine:
        st.markdown("### 🏭 Machine / Opérateur")

        col_m, col_o = st.columns(2)

        with col_m:
            machine_stats = df.groupby("Machine")["Measurement"].agg(["count", "mean", "std"]).reset_index()
            machine_stats.columns = ["Machine", "Nombre", "Moyenne", "Écart-type"]
            st.dataframe(machine_stats, use_container_width=True, hide_index=True)

            fig = px.box(df, x="Machine", y="Measurement", color="Machine", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        with col_o:
            operator_stats = df.groupby("Operator")["Measurement"].agg(["count", "mean", "std"]).reset_index()
            operator_stats.columns = ["Opérateur", "Nombre", "Moyenne", "Écart-type"]
            st.dataframe(operator_stats, use_container_width=True, hide_index=True)

            fig = px.box(df, x="Operator", y="Measurement", color="Operator", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    with tab_interpretation:
        target = (usl + lsl) / 2
        decentrage = mean_val - target

        context_cap = f"""
Moyenne = {mean_val:.4f}
Écart-type = {std_val:.6f}
LS = {usl:.4f}
LI = {lsl:.4f}
Cible = {target:.4f}
Décalage = {decentrage:.6f}
Cp = {cp:.2f}
Cpk = {cpk:.2f}
Nombre de mesures = {total}
"""
        show_ai_analysis("Capabilité", context_cap)


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

        context_pareto = f"""
Défaut principal = {top_defect}
Occurrences défaut principal = {top_count}
Nombre total défauts = {len(defects)}
"""
        show_ai_analysis("Pareto", context_pareto)

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
        if rpn >= 100:
            return "🟡 Élevé"
        return "🟢 Moyen"

    def get_action(rpn):
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
    top_risk = fmea.iloc[0]

    context_amdec = f"""
RPN maximum = {int(max_rpn)}
Défaut principal = {top_risk["Defect_Type"]}
Gravité = {top_risk["Severity"]}
Occurrence = {top_risk["Occurrence"]}
Détection = {top_risk["Detection"]}
Statut = {top_risk["Statut"]}
Action actuelle = {top_risk["Action"]}
"""
    show_ai_analysis("AMDEC", context_amdec)


# =========================
# IA
# =========================
elif page_clean == "IA":
    st.subheader("🤖 Assistant Qualité IA")

    question = st.text_area("Pose ta question qualité")

    if st.button("Analyser"):
        if question.strip() == "":
            st.warning("Écris une question")
        else:
            prompt = f"""
Tu es un expert en qualité industrielle automobile.

Données actuelles :
- Moyenne = {mean_val:.4f}
- Écart-type = {std_val:.6f}
- Cp = {cp:.2f}
- Cpk = {cpk:.2f}
- USL = {usl}
- LSL = {lsl}
- Nombre de mesures = {total}

Question utilisateur :
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


# =========================
# RAPPORT PDF
# =========================
st.markdown("---")
st.subheader("📄 Rapport Qualité")

if st.button("Générer le rapport PDF"):
    pdf_path = generate_pdf_report()

    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📥 Télécharger le rapport PDF",
            data=f,
            file_name="rapport_qualite_specsense.pdf",
            mime="application/pdf"
        )


# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("SpecSense AI V1.0 | Qualité 4.0 | Inspiré IATF 16949")
