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
st.caption("SpecSense AI V1.0 | Qualité 4.0 | Inspiré IATF 16949")    div[data-testid="stMetricValue"] {
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
st.caption("SpecSense AI V1.0 | Qualité 4.0 | Inspiré IATF 16949")    background: linear-gradient(180deg, #020617 0%, #07111f 100%) !important;
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
# AI FUNCTION
# =========================
def ask_hf_ai(question):
    if "HUGGINGFACE_TOKEN" not in st.secrets:
        return "❌ HUGGINGFACE_TOKEN manquant dans Streamlit Secrets."
def generate_ai_module_analysis(module_name, context):
    prompt = f"""
Tu es un expert qualité automobile.

Module analysé : {module_name}

Données :
{context}

Réponds uniquement en français avec ce format :

INTERPRÉTATION :
- ...

ACTIONS RECOMMANDÉES :
- ...
- ...
- ...

Sois clair, professionnel et concret.
"""

    return ask_hf_ai(prompt)
    try:
        client = InferenceClient(token=st.secrets["HUGGINGFACE_TOKEN"])

        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un expert en qualité industrielle automobile. Réponds en français simple avec des actions concrètes."
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


# =========================
# GOOGLE SHEET
# =========================
G_SHEET_URL = "https://docs.google.com/spreadsheets/d/1Xy4tgkGs1OXOTh-OMAsR7YsfkUPxttF7qalhDdhHa90/export?format=csv&gid=0"


@st.cache_data(ttl=60)
def load_data():
    df = pd.read_csv(G_SHEET_URL)
    df.columns = df.columns.str.strip()

    required_cols = [
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
        "Detection"
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"❌ Colonnes manquantes : {missing_cols}")
        st.stop()

    numeric_cols = [
        "Measurement",
        "USL",
        "LSL",
        "Severity",
        "Occurrence",
        "Detection"
    ]

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
# INTERPRETATION AUTO
# =========================
def interpretation_auto(tool):
    if tool == "MSA":
        if len(msa_data) == 0:
            return "Aucune donnée MSA disponible."

        mean_msa = msa_data["Measurement"].mean()
        std_msa = msa_data["Measurement"].std()
        ref = (usl + lsl) / 2
        tolerance = usl - lsl

        cg = (0.2 * tolerance) / (6 * std_msa) if std_msa > 0 else 0
        cgk = (0.1 * tolerance - abs(mean_msa - ref)) / (3 * std_msa) if std_msa > 0 else 0

        if cg >= 1.33 and cgk >= 1.33:
            return f"""
✅ Le système de mesure est acceptable.

Pourquoi ?
- Cg = {cg:.2f} et Cgk = {cgk:.2f}.
- Les deux indicateurs sont supérieurs ou égaux au seuil recommandé 1.33.
- Le système de mesure est stable et maîtrisé.

Actions :
- Maintenir l’étalonnage.
- Continuer la surveillance périodique.
- Standardiser la méthode de mesure.
"""
        return f"""
❌ Le système de mesure n’est pas acceptable.

Pourquoi ?
- Cg = {cg:.2f} et Cgk = {cgk:.2f}.
- Un ou plusieurs indicateurs sont inférieurs au seuil recommandé 1.33.
- Le moyen de mesure ajoute trop de variation ou présente un problème de centrage.

Causes possibles :
- Instrument mal calibré.
- Influence de l’opérateur.
- Méthode non standardisée.
- Conditions de mesure instables.

Actions :
- Recalibrer l’instrument.
- Former les opérateurs.
- Standardiser la méthode.
- Refaire une étude MSA.
"""

    if tool == "SPC":
        mean_spc = spc_data["Measurement"].mean()
        std_spc = spc_data["Measurement"].std()
        ucl = mean_spc + 3 * std_spc
        lcl = mean_spc - 3 * std_spc

        out_control = spc_data[
            (spc_data["Measurement"] > ucl) |
            (spc_data["Measurement"] < lcl)
        ]

        if len(out_control) > 0:
            return f"""
❌ Le processus n’est pas sous contrôle statistique.

Pourquoi ?
- {len(out_control)} point(s) sont hors limites UCL/LCL.
- Cela indique une variation anormale ou une cause spéciale.

Actions :
- Identifier les points hors contrôle.
- Chercher la cause spéciale.
- Corriger immédiatement.
- Vérifier la stabilité après correction.
"""
        return """
✅ Le processus est sous contrôle statistique.

Pourquoi ?
- Tous les points sont dans les limites UCL/LCL.
- Aucune variation anormale n’est détectée.
- Le processus est stable.

Remarque :
- Un processus peut être stable mais non capable si le Cpk est faible.
"""

    if tool == "Capabilité":
        if cpk < 1:
            return f"""
❌ Processus non capable.

Pourquoi ?
- Cpk = {cpk:.2f}, donc il est inférieur à 1.
- Le processus ne respecte pas correctement les tolérances client.
- Il existe un risque de pièces non conformes.

Analyse :
- Cp = {cp:.2f}
- Cpk = {cpk:.2f}
- Si Cp > Cpk, le processus est souvent décentré.

Actions :
- Recentrer le processus.
- Réduire la variation.
- Ajuster les paramètres machine.
- Suivre avec SPC.
"""
        if cpk < 1.33:
            return f"""
⚠️ Processus limite.

Pourquoi ?
- Cpk = {cpk:.2f}, donc il est entre 1 et 1.33.
- La marge de sécurité qualité est faible.

Actions :
- Renforcer la surveillance SPC.
- Réduire la variabilité.
- Stabiliser la machine.
"""
        return f"""
✅ Processus capable.

Pourquoi ?
- Cpk = {cpk:.2f}, donc il est supérieur ou égal à 1.33.
- Le processus respecte les tolérances client.

Actions :
- Maintenir les paramètres actuels.
- Continuer la surveillance.
"""

    if tool == "Pareto":
        defects = df[df["Defect_Type"].astype(str).str.upper() != "OK"]

        if len(defects) > 0:
            top_defect = defects["Defect_Type"].value_counts().idxmax()
            top_count = defects["Defect_Type"].value_counts().max()

            return f"""
🎯 Défaut principal : {top_defect}

Pourquoi ?
- Ce défaut apparaît {top_count} fois.
- C’est le défaut le plus fréquent.
- Selon Pareto, traiter ce défaut peut réduire une grande partie des problèmes qualité.

Actions :
- Identifier la cause racine.
- Lancer une action corrective.
- Suivre ce défaut quotidiennement.
- Vérifier l’efficacité des actions.
"""
        return """
✅ Aucun défaut détecté.

Pourquoi ?
- Toutes les lignes sont OK.
- Aucun défaut principal n’apparaît.

Actions :
- Maintenir les contrôles actuels.
"""

    if tool == "AMDEC":
        rpn_max = (df["Severity"] * df["Occurrence"] * df["Detection"]).max()

        if rpn_max >= 150:
            return f"""
🔴 Risque critique détecté.

Pourquoi ?
- RPN maximum = {int(rpn_max)}, donc il est supérieur ou égal à 150.
- Le risque est critique à cause de la gravité, occurrence ou détection.

Formule :
RPN = Gravité × Occurrence × Détection

Actions :
- Lancer une action immédiate.
- Réduire l’occurrence.
- Améliorer la détection.
- Suivre le RPN après action.
"""
        if rpn_max >= 100:
            return f"""
🟡 Risque élevé.

Pourquoi ?
- RPN maximum = {int(rpn_max)}, donc il est supérieur ou égal à 100.
- Le risque doit être traité.

Actions :
- Mettre une action corrective.
- Renforcer le contrôle.
"""
        return f"""
🟢 Risque acceptable.

Pourquoi ?
- RPN maximum = {int(rpn_max)}, donc il est inférieur à 100.

Actions :
- Maintenir les contrôles.
- Continuer la surveillance.
"""

    return "Aucune interprétation disponible."


# =========================
# PDF
# =========================
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

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

    story.append(Paragraph("Analyse Capabilité", styles["Heading2"]))
    if cpk < 1:
        story.append(Paragraph("[NON] Processus non capable.", styles["BodyText"]))
        story.append(Paragraph("Le processus ne respecte pas les tolérances client.", styles["BodyText"]))
    elif cpk < 1.33:
        story.append(Paragraph("[ATTENTION] Processus limite.", styles["BodyText"]))
        story.append(Paragraph("Une amélioration est nécessaire.", styles["BodyText"]))
    else:
        story.append(Paragraph("[OK] Processus capable.", styles["BodyText"]))
        story.append(Paragraph("Le processus respecte les tolérances.", styles["BodyText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Analyse SPC", styles["Heading2"]))
    mean_spc = spc_data["Measurement"].mean()
    std_spc = spc_data["Measurement"].std()
    ucl = mean_spc + 3 * std_spc
    lcl = mean_spc - 3 * std_spc

    out_control = spc_data[
        (spc_data["Measurement"] > ucl) |
        (spc_data["Measurement"] < lcl)
    ]

    if len(out_control) > 0:
        story.append(Paragraph(f"[NON] {len(out_control)} point(s) hors contrôle.", styles["BodyText"]))
    else:
        story.append(Paragraph("[OK] Processus sous contrôle statistique.", styles["BodyText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Analyse Pareto", styles["Heading2"]))
    defects = df[df["Defect_Type"].astype(str).str.upper() != "OK"]

    if len(defects) > 0:
        top_defect = defects["Defect_Type"].value_counts().idxmax()
        top_count = defects["Defect_Type"].value_counts().max()
        story.append(Paragraph(f"Défaut principal : {top_defect}", styles["BodyText"]))
        story.append(Paragraph(f"Nombre d’occurrences : {top_count}", styles["BodyText"]))
    else:
        story.append(Paragraph("Aucun défaut détecté.", styles["BodyText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Analyse AMDEC", styles["Heading2"]))
    rpn_max = (df["Severity"] * df["Occurrence"] * df["Detection"]).max()
    story.append(Paragraph(f"RPN maximum : {int(rpn_max)}", styles["BodyText"]))

    if rpn_max >= 150:
        story.append(Paragraph("[CRITIQUE] Risque critique.", styles["BodyText"]))
    elif rpn_max >= 100:
        story.append(Paragraph("[ELEVE] Risque élevé.", styles["BodyText"]))
    else:
        story.append(Paragraph("[OK] Risque acceptable.", styles["BodyText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Actions recommandées", styles["Heading2"]))
    if cpk < 1:
        story.append(Paragraph("- Recentrer le processus", styles["BodyText"]))
        story.append(Paragraph("- Réduire la variation", styles["BodyText"]))
        story.append(Paragraph("- Ajuster les paramètres machine", styles["BodyText"]))
    elif cpk < 1.33:
        story.append(Paragraph("- Renforcer le contrôle SPC", styles["BodyText"]))
        story.append(Paragraph("- Stabiliser le processus", styles["BodyText"]))
    else:
        story.append(Paragraph("- Maintenir les conditions actuelles", styles["BodyText"]))

    story.append(Spacer(1, 20))
    story.append(Paragraph("Conclusion", styles["Heading2"]))

    if cpk < 1:
        story.append(Paragraph(
            "Le processus n'est pas conforme aux exigences qualité. "
            "Des actions immédiates sont nécessaires.",
            styles["BodyText"]
        ))
    else:
        story.append(Paragraph(
            "Le processus est globalement maîtrisé.",
            styles["BodyText"]
        ))

    doc = SimpleDocTemplate(doc_path)
    doc.build(story)

    return doc_path
    # =========================
    # SPC
    # =========================
    story.append(Paragraph("Analyse SPC", styles["Heading2"]))

    mean_spc = spc_data["Measurement"].mean()
    std_spc = spc_data["Measurement"].std()

    ucl = mean_spc + 3 * std_spc
    lcl = mean_spc - 3 * std_spc

    out_control = spc_data[
        (spc_data["Measurement"] > ucl) |
        (spc_data["Measurement"] < lcl)
    ]

    if len(out_control) > 0:
        story.append(Paragraph(f"❌ {len(out_control)} point(s) hors contrôle.", styles["BodyText"]))
    else:
        story.append(Paragraph("✅ Processus sous contrôle statistique.", styles["BodyText"]))

    story.append(Spacer(1, 12))

    # =========================
    # PARETO
    # =========================
    story.append(Paragraph("Analyse Pareto", styles["Heading2"]))

    defects = df[df["Defect_Type"].astype(str).str.upper() != "OK"]

    if len(defects) > 0:
        top_defect = defects["Defect_Type"].value_counts().idxmax()
        top_count = defects["Defect_Type"].value_counts().max()

        story.append(Paragraph(f"Défaut principal : {top_defect}", styles["BodyText"]))
        story.append(Paragraph(f"Nombre d’occurrences : {top_count}", styles["BodyText"]))
    else:
        story.append(Paragraph("Aucun défaut détecté.", styles["BodyText"]))

    story.append(Spacer(1, 12))

    # =========================
    # AMDEC
    # =========================
    story.append(Paragraph("Analyse AMDEC", styles["Heading2"]))

    rpn_max = (df["Severity"] * df["Occurrence"] * df["Detection"]).max()

    story.append(Paragraph(f"RPN maximum : {int(rpn_max)}", styles["BodyText"]))

    if rpn_max >= 150:
        story.append(Paragraph("🔴 Risque critique.", styles["BodyText"]))
    elif rpn_max >= 100:
        story.append(Paragraph("🟡 Risque élevé.", styles["BodyText"]))
    else:
        story.append(Paragraph("🟢 Risque acceptable.", styles["BodyText"]))

    story.append(Spacer(1, 12))

    # =========================
    # ACTIONS
    # =========================
    story.append(Paragraph("Actions recommandées", styles["Heading2"]))

    if cpk < 1:
        story.append(Paragraph("- Recentrer le processus", styles["BodyText"]))
        story.append(Paragraph("- Réduire la variation", styles["BodyText"]))
        story.append(Paragraph("- Ajuster les paramètres machine", styles["BodyText"]))
    elif cpk < 1.33:
        story.append(Paragraph("- Renforcer le contrôle SPC", styles["BodyText"]))
        story.append(Paragraph("- Stabiliser le processus", styles["BodyText"]))
    else:
        story.append(Paragraph("- Maintenir les conditions actuelles", styles["BodyText"]))

    # =========================
    # BUILD PDF
    # =========================
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
    <h1 style="margin:0; font-size:44px; font-weight:900;">Tableau de bord</h1>
    <p style="margin:6px 0 0 0; color:#94a3b8; font-size:18px;">
        Vue d’ensemble de la qualité de vos processus
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
# PAGES
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
        st.markdown("### 🎯 Synthèse capabilité")

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

    st.markdown("### 🧠 Synthèse")
    st.info(interpretation_auto("Capabilité"))


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

    # =========================
    # MSA TYPE 1
    # =========================
    with tab_msa1:
        st.markdown("### 📏 MSA Type 1")

        if len(msa_data) > 0:
            mean_msa = msa_data["Measurement"].mean()
            std_msa = msa_data["Measurement"].std()
            ref = (usl + lsl) / 2
            tolerance = usl - lsl

            cg = (0.2 * tolerance) / (6 * std_msa) if std_msa > 0 else 0
            cgk = (0.1 * tolerance - abs(mean_msa - ref)) / (3 * std_msa) if std_msa > 0 else 0

            st.metric("Cg", f"{cg:.2f}")
            st.metric("Cgk", f"{cgk:.2f}")

            st.markdown("### 🤖 Interprétation IA")

            context_msa = f"""
Type analyse : MSA
Référence = {ref:.4f}
Tolérance = {tolerance:.4f}
Cg = {cg:.2f}
Cgk = {cgk:.2f}
Nombre de mesures MSA = {len(msa_data)}
"""

            with st.spinner("🤖 Analyse IA MSA..."):
                ai_msa = generate_ai_module_analysis("MSA", context_msa)

            st.info(ai_msa)

        else:
            st.warning("Aucune donnée MSA disponible.")

    # باقي tabs خليهوم فارغين دابا
    with tab_grr:
        st.info("GRR à ajouter")

    with tab_bias:
        st.info("Bias à ajouter")

    with tab_linearity:
        st.info("Linearity à ajouter")

    with tab_stability:
        st.info("Stability à ajouter")

    with tab_attribute:
        st.info("Attribute MSA à ajouter")

    # =========================
    # MSA TYPE 1
    # =========================
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
            fig.update_layout(
                title="Carte MSA Type 1",
                template="plotly_dark",
                height=430
            )

            st.plotly_chart(fig, use_container_width=True)

            if cg >= 1.33 and cgk >= 1.33:
                st.success("✅ Système de mesure acceptable.")
            else:
                st.error("❌ Système de mesure non acceptable.")

            st.markdown("### 🤖 Interprétation IA")

            context_msa = f"""
Type analyse : MSA Type 1
Référence = {ref:.4f}
Tolérance = {tolerance:.4f}
Moyenne MSA = {mean_msa:.4f}
Écart-type MSA = {std_msa:.6f}
Cg = {cg:.2f}
Cgk = {cgk:.2f}
Nombre de mesures MSA = {len(msa_data)}
"""

            with st.spinner("🤖 Analyse IA MSA..."):
                ai_msa = generate_ai_module_analysis("MSA", context_msa)

            st.info(ai_msa)

        else:
            st.warning("Aucune donnée MSA disponible.")

    with tab_grr:
        st.info("Gage R&R à ajouter ici.")

    with tab_bias:
        st.info("Bias à ajouter ici.")

    with tab_linearity:
        st.info("Linearity à ajouter ici.")

    with tab_stability:
        st.info("Stability à ajouter ici.")

    with tab_attribute:
        st.info("Attribute MSA à ajouter ici.")

    # =========================
    # 2. GAGE R&R
    # =========================
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

            if percent_grr < 10:
                st.success("✅ Gage R&R excellent : système de mesure acceptable.")
            elif percent_grr < 30:
                st.warning("⚠️ Gage R&R acceptable sous conditions.")
            else:
                st.error("❌ Gage R&R non acceptable.")

            fig = px.box(
                df_grr,
                x="Operator",
                y="Measurement",
                color="Operator",
                title="Variation par opérateur",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Aucune donnée MSA disponible.")

    # =========================
    # 3. BIAS
    # =========================
    with tab_bias:
        st.markdown("### 🎯 Bias")

        if len(msa_data) > 0:
            reference = st.number_input("Valeur de référence", value=12.0000, format="%.4f")

            mean_bias = msa_data["Measurement"].mean()
            bias = mean_bias - reference

            c1, c2 = st.columns(2)
            c1.metric("Moyenne mesurée", f"{mean_bias:.6f}")
            c2.metric("Bias", f"{bias:.6f}")

            if abs(bias) <= 0.001:
                st.success("✅ Bias faible : instrument bien centré.")
            else:
                st.error("❌ Bias élevé : instrument décalé par rapport à la référence.")

            st.info("""
Pourquoi ?
- Le Bias représente l'écart entre la moyenne mesurée et la valeur de référence.
- Un Bias élevé indique un problème de calibration ou de centrage.
""")
        else:
            st.warning("Aucune donnée MSA disponible.")

    # =========================
    # 4. LINEARITY
    # =========================
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

            slope_info = df_lin.groupby("Part_ID")["Bias"].mean().std()

            st.metric("Variation du Bias", f"{slope_info:.6f}")

            if slope_info <= 0.001:
                st.success("✅ Linéarité acceptable.")
            else:
                st.warning("⚠️ Linéarité à vérifier : le biais change selon la valeur mesurée.")

            st.info("""
Pourquoi ?
- La linéarité vérifie si l'instrument reste précis sur toute la plage de mesure.
- Si le biais change selon la référence, l'instrument n'est pas linéaire.
""")
        else:
            st.warning("Aucune donnée MSA disponible.")

    # =========================
    # 5. STABILITY
    # =========================
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

            if len(out_stab) == 0:
                st.success("✅ Système stable dans le temps.")
            else:
                st.error(f"❌ {len(out_stab)} point(s) instables détectés.")

        else:
            st.warning("Aucune donnée MSA disponible.")

    # =========================
    # 6. ATTRIBUTE MSA
    # =========================
    with tab_attribute:
        st.markdown("### ✅ Attribute MSA")

        if "Defect_Type" in df.columns:
            df_attr = df.copy()

            df_attr["Decision"] = df_attr["Defect_Type"].astype(str).str.upper().apply(
                lambda x: "OK" if x == "OK" else "NOK"
            )

            total_attr = len(df_attr)
            ok_count = len(df_attr[df_attr["Decision"] == "OK"])
            nok_count = len(df_attr[df_attr["Decision"] == "NOK"])

            agreement = (ok_count / total_attr) * 100 if total_attr > 0 else 0

            c1, c2, c3 = st.columns(3)
            c1.metric("OK", ok_count)
            c2.metric("NOK", nok_count)
            c3.metric("% OK", f"{agreement:.2f}%")

            fig = px.pie(
                df_attr,
                names="Decision",
                title="Répartition OK / NOK",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

            if agreement >= 90:
                st.success("✅ Attribute MSA acceptable.")
            elif agreement >= 75:
                st.warning("⚠️ Attribute MSA moyen : à surveiller.")
            else:
                st.error("❌ Attribute MSA non acceptable.")

            st.info("""
Pourquoi ?
- Attribute MSA est utilisé pour les contrôles visuels ou qualitatifs.
- Il analyse la cohérence des décisions OK / NOK.
""")
        else:
            st.warning("Colonne Defect_Type introuvable.")
elif page_clean == "SPC":
    st.subheader("📉 Module SPC complet")

    tab_control, tab_rules, tab_distribution, tab_capability, tab_machine, tab_interpretation = st.tabs([
        "Carte de contrôle",
        "Règles SPC",
        "Distribution",
        "Capabilité",
        "Machine / Opérateur",
        "Interprétation"
    ])
st.markdown("### 🤖 Interprétation IA")

out_control_count = len(spc_work[spc_work["Hors_Controle"]])

context_spc = f"""
Type analyse : SPC
Moyenne SPC = {mean_spc:.4f}
Écart-type SPC = {std_spc:.6f}
UCL = {ucl:.4f}
LCL = {lcl:.4f}
Points hors contrôle = {out_control_count}
Cp = {cp:.2f}
Cpk = {cpk:.2f}
"""

with st.spinner("🤖 Analyse IA SPC..."):
    ai_spc = generate_ai_module_analysis("SPC", context_spc)

st.info(ai_spc)
    mean_spc = spc_data["Measurement"].mean()
    std_spc = spc_data["Measurement"].std()

    ucl = mean_spc + 3 * std_spc
    lcl = mean_spc - 3 * std_spc

    spc_work = spc_data.copy().reset_index(drop=True)
    spc_work["Point"] = range(1, len(spc_work) + 1)
    spc_work["CL"] = mean_spc
    spc_work["UCL"] = ucl
    spc_work["LCL"] = lcl
    spc_work["Hors_Controle"] = (
        (spc_work["Measurement"] > ucl) |
        (spc_work["Measurement"] < lcl)
    )

    # =========================
    # 1. CARTE DE CONTRÔLE
    # =========================
    with tab_control:
        st.markdown("### 📈 Carte de contrôle")

        c1, c2, c3 = st.columns(3)
        c1.metric("Ligne centrale", f"{mean_spc:.4f}")
        c2.metric("Limite supérieure UCL", f"{ucl:.4f}")
        c3.metric("Limite inférieure LCL", f"{lcl:.4f}")

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

        fig.update_layout(
            title="Carte de contrôle SPC",
            template="plotly_dark",
            height=460,
            xaxis_title="Point",
            yaxis_title="Mesure"
        )

        st.plotly_chart(fig, use_container_width=True)

        out_control = spc_work[spc_work["Hors_Controle"]]

        if len(out_control) > 0:
            st.error(f"❌ {len(out_control)} point(s) hors limites de contrôle.")
            st.dataframe(out_control, use_container_width=True, hide_index=True)
        else:
            st.success("✅ Aucun point hors limites de contrôle.")

    # =========================
    # 2. RÈGLES SPC
    # =========================
    with tab_rules:
        st.markdown("### 🚦 Règles SPC automatiques")

        out_control = spc_work[spc_work["Hors_Controle"]]

        # Règle 1 : point hors limites
        rule1 = len(out_control)

        # Règle 2 : 7 points consécutifs du même côté de la moyenne
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

        # Règle 3 : tendance 6 points montants ou descendants
        trend_detected = False

        for i in range(len(values) - 5):
            segment = values[i:i + 6]

            increasing = all(segment[j] < segment[j + 1] for j in range(5))
            decreasing = all(segment[j] > segment[j + 1] for j in range(5))

            if increasing or decreasing:
                trend_detected = True
                break

        r1, r2, r3 = st.columns(3)

        if rule1 > 0:
            r1.error(f"❌ Règle 1: {rule1} point(s) hors contrôle")
        else:
            r1.success("✅ Règle 1: OK")

        if rule2:
            r2.warning("⚠️ Règle 2: série de 7 points détectée")
        else:
            r2.success("✅ Règle 2: OK")

        if trend_detected:
            r3.warning("⚠️ Règle 3: tendance détectée")
        else:
            r3.success("✅ Règle 3: OK")

        st.markdown("### 🧠 Signification")

        st.info("""
Règles utilisées :
- Règle 1 : un point hors UCL/LCL indique une cause spéciale.
- Règle 2 : 7 points du même côté de la moyenne indiquent un décalage du processus.
- Règle 3 : 6 points montants ou descendants indiquent une tendance ou dérive.
""")

    # =========================
    # 3. DISTRIBUTION
    # =========================
    with tab_distribution:
        st.markdown("### 📊 Distribution des mesures")

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

        skew = spc_work["Measurement"].skew()

        st.metric("Asymétrie distribution", f"{skew:.3f}")

        if abs(skew) < 0.5:
            st.success("✅ Distribution globalement équilibrée.")
        else:
            st.warning("⚠️ Distribution asymétrique : vérifier centrage ou causes spéciales.")

    # =========================
    # 4. CAPABILITÉ DANS SPC
    # =========================
    with tab_capability:
        st.markdown("### 🎯 Capabilité SPC")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("USL", f"{usl:.4f}")
        c2.metric("LSL", f"{lsl:.4f}")
        c3.metric("Cp", f"{cp:.2f}")
        c4.metric("Cpk", f"{cpk:.2f}")

        if cpk >= 1.33:
            st.success("✅ Processus capable.")
        elif cpk >= 1:
            st.warning("⚠️ Processus limite.")
        else:
            st.error("❌ Processus non capable.")

        st.info("""
Cp mesure le potentiel du processus.
Cpk mesure la capabilité réelle en tenant compte du centrage.
Si Cp > Cpk, le processus est souvent décentré.
""")

    # =========================
    # 5. MACHINE / OPÉRATEUR
    # =========================
    with tab_machine:
        st.markdown("### 🏭 Analyse Machine / Opérateur")

        col_m, col_o = st.columns(2)

        with col_m:
            if "Machine" in spc_work.columns:
                machine_stats = spc_work.groupby("Machine")["Measurement"].agg(["count", "mean", "std"]).reset_index()
                machine_stats.columns = ["Machine", "Nombre", "Moyenne", "Écart-type"]

                st.markdown("#### Machine")
                st.dataframe(machine_stats, use_container_width=True, hide_index=True)

                fig = px.box(
                    spc_work,
                    x="Machine",
                    y="Measurement",
                    color="Machine",
                    title="Variation par machine",
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Colonne Machine introuvable.")

        with col_o:
            if "Operator" in spc_work.columns:
                operator_stats = spc_work.groupby("Operator")["Measurement"].agg(["count", "mean", "std"]).reset_index()
                operator_stats.columns = ["Opérateur", "Nombre", "Moyenne", "Écart-type"]

                st.markdown("#### Opérateur")
                st.dataframe(operator_stats, use_container_width=True, hide_index=True)

                fig = px.box(
                    spc_work,
                    x="Operator",
                    y="Measurement",
                    color="Operator",
                    title="Variation par opérateur",
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Colonne Operator introuvable.")

    # =========================
    # 6. INTERPRÉTATION
    # =========================
    with tab_interpretation:
        st.markdown("### 🧠 Interprétation SPC détaillée")

        out_control = spc_work[spc_work["Hors_Controle"]]

        if len(out_control) > 0:
            st.error(f"""
❌ Le processus présente {len(out_control)} point(s) hors contrôle.

Pourquoi ?
- Un ou plusieurs points dépassent les limites UCL/LCL.
- Cela indique une cause spéciale ou une variation anormale.

Actions :
- Identifier les points concernés.
- Vérifier machine, matière, méthode et opérateur.
- Corriger la cause spéciale avant de continuer la production.
""")
        else:
            st.success("""
✅ Le processus est sous contrôle statistique.

Pourquoi ?
- Tous les points sont à l'intérieur des limites UCL/LCL.
- Aucune cause spéciale majeure n'est détectée.
- La variation observée est une variation normale du processus.
""")

        if cpk < 1:
            st.error(f"""
❌ Attention : le processus est stable mais non capable.

Pourquoi ?
- Cpk = {cpk:.2f}, inférieur à 1.
- Le processus peut être stable statistiquement, mais ne respecte pas bien les tolérances client.

Actions :
- Recentrer le processus.
- Réduire la variation.
- Ajuster les paramètres machine.
""")
        elif cpk < 1.33:
            st.warning(f"""
⚠️ Processus limite.

Pourquoi ?
- Cpk = {cpk:.2f}, inférieur au seuil recommandé 1.33.
- Le risque qualité existe encore.

Actions :
- Renforcer le suivi SPC.
- Réduire la variabilité.
""")
        else:
            st.success(f"""
✅ Processus capable.

Pourquoi ?
- Cpk = {cpk:.2f}, supérieur ou égal à 1.33.
- Le processus respecte les tolérances client.
""")

elif page_clean == "Capabilité":
    st.subheader("🎯 Module Capabilité complet")

    tab_kpi, tab_hist, tab_centering, tab_machine, tab_interpretation = st.tabs([
        "Indices Cp / Cpk",
        "Histogramme",
        "Centrage",
        "Machine / Opérateur",
        "Interprétation"
    ])
st.markdown("### 🤖 Interprétation IA")

target = (usl + lsl) / 2
decentrage = mean_val - target

context_cap = f"""
Type analyse : Capabilité
Moyenne = {mean_val:.4f}
Écart-type = {std_val:.6f}
USL / LS = {usl:.4f}
LSL / LI = {lsl:.4f}
Cible = {target:.4f}
Décalage = {decentrage:.6f}
Cp = {cp:.2f}
Cpk = {cpk:.2f}
Nombre de mesures = {total}
"""

with st.spinner("🤖 Analyse IA Capabilité..."):
    ai_cap = generate_ai_module_analysis("Capabilité", context_cap)

st.info(ai_cap)
    # =========================
    # 1. INDICES CP / CPK
    # =========================
    with tab_kpi:
        st.markdown("### 📌 Indices de capabilité")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("LS", f"{usl:.4f}")
        c2.metric("LI", f"{lsl:.4f}")
        c3.metric("Cp", f"{cp:.2f}")
        c4.metric("Cpk", f"{cpk:.2f}")

        st.info("""
Cp mesure la capacité théorique du processus.

Cpk mesure la capacité réelle en tenant compte du centrage.
Si Cp > Cpk, le processus est généralement décentré.
""")

        if cpk >= 1.67:
            st.success("🟢 Capabilité excellente.")
        elif cpk >= 1.33:
            st.success("✅ Processus capable.")
        elif cpk >= 1.00:
            st.warning("⚠️ Processus limite.")
        else:
            st.error("❌ Processus non capable.")

    # =========================
    # 2. HISTOGRAMME
    # =========================
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

        out_spec = df[
            (df["Measurement"] > usl) |
            (df["Measurement"] < lsl)
        ]

        if len(out_spec) > 0:
            st.error(f"❌ {len(out_spec)} mesure(s) hors spécifications.")
            st.dataframe(out_spec, use_container_width=True, hide_index=True)
        else:
            st.success("✅ Toutes les mesures sont dans les spécifications.")

    # =========================
    # 3. CENTRAGE
    # =========================
    with tab_centering:
        st.markdown("### 🎯 Analyse du centrage")

        target = (usl + lsl) / 2
        decentrage = mean_val - target

        c1, c2, c3 = st.columns(3)
        c1.metric("Cible", f"{target:.4f}")
        c2.metric("Moyenne", f"{mean_val:.4f}")
        c3.metric("Décalage", f"{decentrage:.6f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(df) + 1)),
            y=df["Measurement"],
            mode="lines+markers",
            name="Mesures"
        ))

        fig.add_hline(y=target, line_dash="solid", annotation_text="Cible")
        fig.add_hline(y=mean_val, line_dash="dash", annotation_text="Moyenne")
        fig.add_hline(y=usl, line_dash="dot", annotation_text="LS")
        fig.add_hline(y=lsl, line_dash="dot", annotation_text="LI")

        fig.update_layout(
            title="Centrage du processus",
            template="plotly_dark",
            height=430
        )

        st.plotly_chart(fig, use_container_width=True)

        if abs(decentrage) <= (usl - lsl) * 0.05:
            st.success("✅ Processus bien centré.")
        else:
            st.warning("⚠️ Processus décentré. Ajustement machine recommandé.")

    # =========================
    # 4. MACHINE / OPÉRATEUR
    # =========================
    with tab_machine:
        st.markdown("### 🏭 Capabilité par Machine / Opérateur")

        col_m, col_o = st.columns(2)

        with col_m:
            if "Machine" in df.columns:
                machine_stats = df.groupby("Machine")["Measurement"].agg(["count", "mean", "std"]).reset_index()
                machine_stats.columns = ["Machine", "Nombre", "Moyenne", "Écart-type"]

                st.markdown("#### Machine")
                st.dataframe(machine_stats, use_container_width=True, hide_index=True)

                fig = px.box(
                    df,
                    x="Machine",
                    y="Measurement",
                    color="Machine",
                    title="Distribution par machine",
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Colonne Machine introuvable.")

        with col_o:
            if "Operator" in df.columns:
                operator_stats = df.groupby("Operator")["Measurement"].agg(["count", "mean", "std"]).reset_index()
                operator_stats.columns = ["Opérateur", "Nombre", "Moyenne", "Écart-type"]

                st.markdown("#### Opérateur")
                st.dataframe(operator_stats, use_container_width=True, hide_index=True)

                fig = px.box(
                    df,
                    x="Operator",
                    y="Measurement",
                    color="Operator",
                    title="Distribution par opérateur",
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Colonne Operator introuvable.")

    # =========================
    # 5. INTERPRÉTATION
    # =========================
    with tab_interpretation:
        st.markdown("### 🧠 Interprétation Capabilité détaillée")

        target = (usl + lsl) / 2
        decentrage = mean_val - target

        if cpk < 1:
            st.error(f"""
❌ Processus non capable.

Pourquoi ?
- Cpk = {cpk:.2f}, donc inférieur à 1.
- Le processus ne respecte pas correctement les tolérances client.
- Il existe un risque élevé de pièces non conformes.

Analyse :
- Cp = {cp:.2f}
- Cpk = {cpk:.2f}
- Décalage moyenne/cible = {decentrage:.6f}

Causes possibles :
- Processus décentré
- Variation trop élevée
- Réglage machine incorrect
- Usure outil ou instabilité process

Actions :
- Recentrer le processus
- Réduire la variation
- Ajuster les paramètres machine
- Renforcer le suivi SPC
""")

        elif cpk < 1.33:
            st.warning(f"""
⚠️ Processus limite.

Pourquoi ?
- Cpk = {cpk:.2f}, inférieur au seuil recommandé 1.33.
- Le processus produit globalement bien, mais avec une marge qualité faible.

Actions :
- Réduire la variabilité
- Surveiller les dérives SPC
- Stabiliser les conditions de production
""")

        else:
            st.success(f"""
✅ Processus capable.

Pourquoi ?
- Cpk = {cpk:.2f}, supérieur ou égal à 1.33.
- Le processus respecte les tolérances client avec une marge suffisante.

Actions :
- Maintenir les paramètres actuels
- Continuer la surveillance SPC
- Standardiser les bonnes pratiques
""")


elif page_clean == "Pareto":
    st.subheader("📊 Analyse Pareto des défauts")

    defects = df[df["Defect_Type"].astype(str).str.upper() != "OK"]

    if len(defects) > 0:
        pareto = defects["Defect_Type"].value_counts().reset_index()
        pareto.columns = ["Type de défaut", "Nombre"]
        pareto["Cumul %"] = pareto["Nombre"].cumsum() / pareto["Nombre"].sum() * 100

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=pareto["Type de défaut"],
            y=pareto["Nombre"],
            name="Défauts"
        ))

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

    else:
        st.success("✅ Aucun défaut détecté.")

    st.markdown("### 🧠 Interprétation détaillée")
    st.info(interpretation_auto("Pareto"))


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
        "Part_ID",
        "Defect_Type",
        "Severity",
        "Occurrence",
        "Detection",
        "RPN",
        "Statut",
        "Action"
    ]].rename(columns={
        "Part_ID": "Référence pièce",
        "Defect_Type": "Type de défaut",
        "Severity": "Gravité",
        "Detection": "Détection"
    })
st.markdown("### 🤖 Interprétation IA")

max_rpn = fmea["RPN"].max()
top_risk = fmea.iloc[0]

context_amdec = f"""
Type analyse : AMDEC
RPN maximum = {int(max_rpn)}
Défaut principal = {top_risk["Defect_Type"]}
Gravité = {top_risk["Severity"]}
Occurrence = {top_risk["Occurrence"]}
Détection = {top_risk["Detection"]}
Statut = {top_risk["Statut"]}
Action actuelle = {top_risk["Action"]}
"""

with st.spinner("🤖 Analyse IA AMDEC..."):
    ai_amdec = generate_ai_module_analysis("AMDEC", context_amdec)

st.info(ai_amdec)
    st.dataframe(table_fmea, use_container_width=True, hide_index=True)

    st.markdown("### 🧠 Interprétation détaillée")
    st.info(interpretation_auto("AMDEC"))


elif page_clean == "IA":
    st.subheader("🤖 Assistant Qualité IA")

    question = st.text_area("Pose ta question qualité")

    if st.button("Analyser"):
        if question.strip() == "":
            st.warning("Écris une question")
        else:
            prompt = f"""
Tu es un expert en qualité industrielle automobile.

Réponds en français simple et professionnel.

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
