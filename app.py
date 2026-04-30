import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from huggingface_hub import InferenceClient
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


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
# STYLE CSS
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
# GOOGLE SHEET
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

    missing_cols = [c for c in required_cols if c not in df.columns]
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
# IA
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
def generate_pdf_report():
    doc_path = "rapport_qualite_specsense.pdf"
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Rapport Qualité - SpecSense AI", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Résumé global", styles["Heading2"]))
    story.append(Paragraph(f"Nombre total de mesures : {total}", styles["BodyText"]))
    story.append(Paragraph(f"Points MSA : {msa_count}", styles["BodyText"]))
    story.append(Paragraph(f"Points SPC : {spc_count}", styles["BodyText"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Indicateurs processus", styles["Heading2"]))
    story.append(Paragraph(f"Moyenne : {mean_val:.4f}", styles["BodyText"]))
    story.append(Paragraph(f"Écart-type : {std_val:.6f}", styles["BodyText"]))
    story.append(Paragraph(f"Cp : {cp:.2f}", styles["BodyText"]))
    story.append(Paragraph(f"Cpk : {cpk:.2f}", styles["BodyText"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Conclusion", styles["Heading2"]))

    if cpk >= 1.33:
        conclusion = "Le processus est capable de respecter les tolérances client."
    elif cpk >= 1:
        conclusion = "Le processus est limite. Une amélioration est nécessaire."
    else:
        conclusion = "Le processus n’est pas capable. Des actions correctives sont requises."

    story.append(Paragraph(conclusion, styles["BodyText"]))

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
            "🤖 IA",
        ],
        label_visibility="collapsed",
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
            name="Mesures",
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
            title="Distribution des mesures",
        )

        fig.add_vline(x=usl, line_dash="dash", annotation_text="USL")
        fig.add_vline(x=lsl, line_dash="dash", annotation_text="LSL")
        fig.add_vline(x=mean_val, line_dash="dot", annotation_text="Moyenne")

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧠 Synthèse")
    st.info(interpretation_auto("Capabilité"))


# =========================
# MSA
# =========================
elif page_clean == "MSA":
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
            name="Mesures MSA",
        ))

        fig.add_hline(y=mean_msa, line_dash="dash", annotation_text="Moyenne")
        fig.add_hline(y=ref, line_dash="dot", annotation_text="Référence")
        fig.update_layout(title="Carte MSA", template="plotly_dark", height=430)

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(msa_data, use_container_width=True, hide_index=True)

        st.markdown("### 🧠 Interprétation détaillée")
        st.info(interpretation_auto("MSA"))

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
        name="SPC",
    ))

    fig.add_hline(y=mean_spc, line_dash="dash", annotation_text="CL")
    fig.add_hline(y=ucl, line_dash="dash", annotation_text="UCL")
    fig.add_hline(y=lcl, line_dash="dash", annotation_text="LCL")
    fig.add_hline(y=usl, line_dash="dot", annotation_text="USL")
    fig.add_hline(y=lsl, line_dash="dot", annotation_text="LSL")

    fig.update_layout(title="Carte de contrôle", template="plotly_dark", height=460)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧠 Interprétation détaillée")
    st.info(interpretation_auto("SPC"))


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
        template="plotly_dark",
    )

    fig.add_vline(x=usl, line_dash="dash", annotation_text="USL")
    fig.add_vline(x=lsl, line_dash="dash", annotation_text="LSL")
    fig.add_vline(x=mean_val, line_dash="dot", annotation_text="Moyenne")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧠 Interprétation détaillée")
    st.info(interpretation_auto("Capabilité"))


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

        fig.add_trace(go.Bar(
            x=pareto["Type de défaut"],
            y=pareto["Nombre"],
            name="Défauts",
        ))

        fig.add_trace(go.Scatter(
            x=pareto["Type de défaut"],
            y=pareto["Cumul %"],
            yaxis="y2",
            mode="lines+markers",
            name="Cumul %",
        ))

        fig.update_layout(
            title="Diagramme Pareto",
            template="plotly_dark",
            height=460,
            yaxis=dict(title="Nombre"),
            yaxis2=dict(title="Cumul %", overlaying="y", side="right", range=[0, 110]),
        )

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pareto, use_container_width=True, hide_index=True)

    else:
        st.success("✅ Aucun défaut détecté.")

    st.markdown("### 🧠 Interprétation détaillée")
    st.info(interpretation_auto("Pareto"))


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
        "Part_ID",
        "Defect_Type",
        "Severity",
        "Occurrence",
        "Detection",
        "RPN",
        "Statut",
        "Action",
    ]].rename(columns={
        "Part_ID": "Référence pièce",
        "Defect_Type": "Type de défaut",
        "Severity": "Gravité",
        "Detection": "Détection",
    })

    st.dataframe(table_fmea, use_container_width=True, hide_index=True)

    st.markdown("### 🧠 Interprétation détaillée")
    st.info(interpretation_auto("AMDEC"))


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
            mime="application/pdf",
        )


# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("SpecSense AI V1.0 | Qualité 4.0 | Inspiré IATF 16949")
                    "role": "system",
                    "content": (
                        "Tu es un expert en qualité industrielle automobile. "
                        "Réponds en français simple, professionnel, avec des actions concrètes."
                    ),
                },
                {"role": "user", "content": question},
            ],
            max_tokens=600,
            temperature=0.3,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Erreur IA : {e}"


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
<style>

div[data-testid="stMetricLabel"] p {
    color: #cbd5e1 !important;
    font-weight: 700 !important;
    font-size: 16px !important;
}

</style>
""", unsafe_allow_html=True)
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
        "Detection",
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
        "Detection",
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
        (mean_val - lsl) / (3 * std_val),
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
- Cg = {cg:.2f} et Cgk = {cgk:.2f}, donc ils sont supérieurs ou égaux au seuil recommandé 1.33.
- La variation du moyen de mesure est maîtrisée.
- Le système de mesure peut être utilisé pour suivre le processus.

Actions :
- Maintenir l’étalonnage.
- Continuer la surveillance périodique.
- Standardiser la méthode de mesure.
"""
        else:
            return f"""
❌ Le système de mesure n’est pas acceptable.

Pourquoi ?
- Cg = {cg:.2f} et Cgk = {cgk:.2f}.
- Un ou plusieurs indicateurs sont inférieurs au seuil recommandé 1.33.
- Cela signifie que le moyen de mesure ajoute trop de variation ou présente un problème de centrage.

Causes possibles :
- Instrument mal calibré.
- Influence de l’opérateur.
- Méthode de mesure non standardisée.
- Conditions de mesure instables.

Actions :
- Recalibrer l’instrument.
- Former les opérateurs.
- Standardiser la méthode de mesure.
- Refaire une étude MSA après correction.
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
- {len(out_control)} point(s) sont en dehors des limites de contrôle UCL/LCL.
- Cela indique une cause spéciale ou une variation anormale.

Causes possibles :
- Réglage machine instable.
- Changement matière.
- Erreur opérateur.
- Usure outil ou problème maintenance.

Actions :
- Identifier les points hors contrôle.
- Chercher la cause spéciale.
- Corriger immédiatement.
- Confirmer la stabilité après correction.
"""
        else:
            return f"""
✅ Le processus est sous contrôle statistique.

Pourquoi ?
- Tous les points sont dans les limites de contrôle UCL/LCL.
- Aucune variation anormale n’est détectée.
- Le processus est stable statistiquement.

Remarque importante :
- Un processus peut être stable mais non capable si le Cpk est faible.
- Ici, la stabilité SPC ne garantit pas automatiquement le respect des tolérances client.

Actions :
- Continuer la surveillance.
- Vérifier la capabilité Cpk.
- Maintenir les paramètres actuels.
"""

    if tool == "Capabilité":
        if cpk < 1:
            return f"""
❌ Processus non capable.

Pourquoi ?
- Cpk = {cpk:.2f}, donc il est inférieur à 1.
- Le processus ne respecte pas correctement les tolérances client.
- Il existe un risque important de produire des pièces non conformes.

Analyse :
- Cp = {cp:.2f}
- Cpk = {cpk:.2f}
- Si Cp est supérieur à Cpk, cela signifie souvent que le processus est décentré.

Actions :
- Recentrer le processus autour de la cible.
- Réduire la variation.
- Ajuster les paramètres machine.
- Suivre l’évolution avec SPC.
"""
        elif cpk < 1.33:
            return f"""
⚠️ Processus limite.

Pourquoi ?
- Cpk = {cpk:.2f}, donc il est entre 1 et 1.33.
- Le processus fonctionne, mais la marge de sécurité qualité est faible.

Actions :
- Renforcer la surveillance SPC.
- Réduire la variabilité.
- Stabiliser les paramètres machine.
"""
        else:
            return f"""
✅ Processus capable.

Pourquoi ?
- Cpk = {cpk:.2f}, donc il est supérieur ou égal à 1.33.
- Le processus respecte les tolérances client avec une marge suffisante.

Actions :
- Maintenir les paramètres actuels.
- Continuer la surveillance qualité.
- Standardiser les bonnes pratiques.
"""

    if tool == "Pareto":
        defects = df[df["Defect_Type"].astype(str).str.upper() != "OK"]

        if len(defects) > 0:
            top_defect = defects["Defect_Type"].value_counts().idxmax()
            top_count = defects["Defect_Type"].value_counts().max()

            return f"""
🎯 Défaut principal : {top_defect}

Pourquoi ?
- Le défaut {top_defect} apparaît {top_count} fois.
- C’est le défaut le plus fréquent dans les données.
- Selon Pareto, traiter le défaut principal peut réduire une grande partie des problèmes qualité.

Causes possibles :
- Problème machine.
- Mauvaise manipulation.
- Défaut matière.
- Problème de stockage ou transport.

Actions :
- Identifier la cause racine.
- Lancer une action corrective.
- Suivre ce défaut quotidiennement.
- Vérifier l’efficacité des actions.
"""
        return """
✅ Aucun défaut détecté.

Pourquoi ?
- Toutes les lignes sont marquées OK.
- Aucun défaut principal n’apparaît dans les données.

Actions :
- Maintenir les contrôles actuels.
- Continuer le suivi qualité.
"""

    if tool == "AMDEC":
        rpn_max = (df["Severity"] * df["Occurrence"] * df["Detection"]).max()

        if rpn_max >= 150:
            return f"""
🔴 Risque critique détecté.

Pourquoi ?
- Le RPN maximum = {int(rpn_max)}, donc il est supérieur ou égal à 150.
- Le risque est critique car la gravité, l’occurrence ou la détection sont élevées.

Formule :
RPN = Gravité × Occurrence × Détection

Actions :
- Lancer une action immédiate.
- Réduire l’occurrence du défaut.
- Améliorer la détection.
- Mettre un plan d’action correctif.
- Suivre le RPN après action.
"""
        elif rpn_max >= 100:
            return f"""
🟡 Risque élevé.

Pourquoi ?
- Le RPN maximum = {int(rpn_max)}, donc il est supérieur ou égal à 100.
- Le risque doit être traité avant qu’il devienne critique.

Actions :
- Mettre en place une action corrective.
- Renforcer le contrôle.
- Réduire la fréquence d’apparition.
"""
        return f"""
🟢 Risque acceptable.

Pourquoi ?
- Le RPN maximum = {int(rpn_max)}, donc il est inférieur à 100.
- Le niveau de risque reste maîtrisé.

Actions :
- Maintenir les contrôles.
- Continuer la surveillance.
"""

    return "Aucune interprétation disponible."


# =========================
# PDF
# =========================
def generate_pdf_report():
    doc_path = "rapport_qualite_specsense.pdf"
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Rapport Qualité - SpecSense AI", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Résumé global", styles["Heading2"]))
    story.append(Paragraph(f"Nombre total de mesures : {total}", styles["BodyText"]))
    story.append(Paragraph(f"Points MSA : {msa_count}", styles["BodyText"]))
    story.append(Paragraph(f"Points SPC : {spc_count}", styles["BodyText"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Indicateurs processus", styles["Heading2"]))
    story.append(Paragraph(f"Moyenne : {mean_val:.4f}", styles["BodyText"]))
    story.append(Paragraph(f"Écart-type : {std_val:.6f}", styles["BodyText"]))
    story.append(Paragraph(f"Cp : {cp:.2f}", styles["BodyText"]))
    story.append(Paragraph(f"Cpk : {cpk:.2f}", styles["BodyText"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Conclusion", styles["Heading2"]))

    if cpk >= 1.33:
        conclusion = "Le processus est capable de respecter les tolérances client."
    elif cpk >= 1:
        conclusion = "Le processus est limite. Une amélioration est nécessaire."
    else:
        conclusion = "Le processus n’est pas capable. Des actions correctives sont requises."

    story.append(Paragraph(conclusion, styles["BodyText"]))

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
            "🤖 IA",
        ],
        label_visibility="collapsed",
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
            name="Mesures",
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
            title="Distribution des mesures",
        )

        fig.add_vline(x=usl, line_dash="dash", annotation_text="USL")
        fig.add_vline(x=lsl, line_dash="dash", annotation_text="LSL")
        fig.add_vline(x=mean_val, line_dash="dot", annotation_text="Moyenne")

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧠 Synthèse")

    if cpk >= 1.33:
        st.success("✅ Le processus est capable et stable.")
    elif cpk >= 1:
        st.warning("⚠️ Le processus est limite. Une amélioration est nécessaire.")
    else:
        st.error("❌ Le processus n’est pas capable. Actions correctives requises.")


# =========================
# MSA
# =========================
elif page_clean == "MSA":
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
            name="Mesures MSA",
        ))

        fig.add_hline(y=mean_msa, line_dash="dash", annotation_text="Moyenne")
        fig.add_hline(y=ref, line_dash="dot", annotation_text="Référence")
        fig.update_layout(title="Carte MSA", template="plotly_dark", height=430)

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(msa_data, use_container_width=True, hide_index=True)

        st.markdown("### 🧠 Interprétation détaillée")
        st.info(interpretation_auto("MSA"))

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
        name="SPC",
    ))

    fig.add_hline(y=mean_spc, line_dash="dash", annotation_text="CL")
    fig.add_hline(y=ucl, line_dash="dash", annotation_text="UCL")
    fig.add_hline(y=lcl, line_dash="dash", annotation_text="LCL")
    fig.add_hline(y=usl, line_dash="dot", annotation_text="USL")
    fig.add_hline(y=lsl, line_dash="dot", annotation_text="LSL")

    fig.update_layout(title="Carte de contrôle", template="plotly_dark", height=460)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧠 Interprétation détaillée")
    st.info(interpretation_auto("SPC"))


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
        template="plotly_dark",
    )

    fig.add_vline(x=usl, line_dash="dash", annotation_text="USL")
    fig.add_vline(x=lsl, line_dash="dash", annotation_text="LSL")
    fig.add_vline(x=mean_val, line_dash="dot", annotation_text="Moyenne")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧠 Interprétation détaillée")
    st.info(interpretation_auto("Capabilité"))


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

        fig.add_trace(go.Bar(
            x=pareto["Type de défaut"],
            y=pareto["Nombre"],
            name="Défauts",
        ))

        fig.add_trace(go.Scatter(
            x=pareto["Type de défaut"],
            y=pareto["Cumul %"],
            yaxis="y2",
            mode="lines+markers",
            name="Cumul %",
        ))

        fig.update_layout(
            title="Diagramme Pareto",
            template="plotly_dark",
            height=460,
            yaxis=dict(title="Nombre"),
            yaxis2=dict(title="Cumul %", overlaying="y", side="right", range=[0, 110]),
        )

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pareto, use_container_width=True, hide_index=True)

        st.markdown("### 🧠 Interprétation détaillée")
        st.info(interpretation_auto("Pareto"))

    else:
        st.success("✅ Aucun défaut détecté.")
        st.markdown("### 🧠 Interprétation détaillée")
        st.info(interpretation_auto("Pareto"))


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
        return "🟢 Moyen"

    def get_action(rpn):
        if rpn >= 150:
            return "Action immédiate requise"
        elif rpn >= 100:
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
        "Action",
    ]].rename(columns={
        "Part_ID": "Référence pièce",
        "Defect_Type": "Type de défaut",
        "Severity": "Gravité",
        "Detection": "Détection",
    })

    st.dataframe(table_fmea, use_container_width=True, hide_index=True)

    st.markdown("### 🧠 Interprétation détaillée")
    st.info(interpretation_auto("AMDEC"))


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
            mime="application/pdf",
        )


# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("SpecSense AI V1.0 | Qualité 4.0 | Inspiré IATF 16949")color: #cbd5e1 !important;
    font-weight: 700 !important;
    
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
