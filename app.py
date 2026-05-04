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
G_SCRIPT_URL = "https://script.google.com/macros/s/AKfycbxchzK3FgKprug2PjVZd_LtuZeErsFdcZoQMIOZGttKOHigXoCCPtVnD_46Cm2nxeVVyA/exec"
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
def inject_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(135deg, #020617 0%, #07111f 45%, #0b1220 100%);
                color: white;
            }

            .block-container {
                padding-top: 2rem;
                padding-left: 2rem;
                padding-right: 2rem;
            }

            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #020617 0%, #07111f 100%) !important;
                border-right: 1px solid rgba(255,255,255,.08);
            }

            div[data-testid="stMetric"] {
                background: radial-gradient(circle at top left, rgba(0,212,255,.18), rgba(15,23,42,.92));
                border: 1px solid rgba(0,212,255,.45);
                border-radius: 22px;
                padding: 20px;
                box-shadow: 0 18px 40px rgba(0,0,0,.25);
            }

            div[data-testid="stMetricValue"] {
                color: white !important;
                font-size: 34px !important;
                font-weight: 900 !important;
            }

            div[role="radiogroup"] label {
                background: rgba(15,23,42,.65);
                border: 1px solid rgba(148,163,184,.14);
                border-radius: 14px;
                padding: 10px 12px;
                margin-bottom: 8px;
            }

            @media (max-width: 768px) {
                .block-container {
                    padding: 1rem !important;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
    # st.stop()    st.stop()

    for col in NUMERIC_COLS:
        df[col] = df[col].astype(str).str.replace(",", ".", regex=False).str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid_rows = df[df[NUMERIC_COLS].isna().any(axis=1)]
    if not invalid_rows.empty:
        st.error("❌ Erreur data : certaines valeurs numériques sont invalides.")
        st.dataframe(invalid_rows, use_container_width=True)
        st.stop()

    return df


def save_to_google_sheet(row: dict) -> None:
    try:
        requests.post(
            G_SHEET_URL.replace("output=csv", "exec"),
            json=row,
            timeout=5
        )
    except Exception:
        pass

def prepare_data(df: pd.DataFrame) -> dict:
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
        cpk = min((usl - mean_val) / (3 * std_val), (mean_val - lsl) / (3 * std_val))
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
def generate_pdf_report(metrics: dict) -> str:
    styles = getSampleStyleSheet()
    story = []

    if os.path.exists(LOGO_PATH):
        story.append(Image(LOGO_PATH, width=2 * inch, height=1 * inch))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Rapport Qualité - SpecSense AI", styles["Title"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Indicateurs clés", styles["Heading2"]))
    story.append(Paragraph(f"Nombre total de mesures : {metrics['total']}", styles["BodyText"]))
    story.append(Paragraph(f"Moyenne : {metrics['mean_val']:.4f}", styles["BodyText"]))
    story.append(Paragraph(f"Écart-type : {metrics['std_val']:.6f}", styles["BodyText"]))
    story.append(Paragraph(f"Cp : {metrics['cp']:.2f}", styles["BodyText"]))
    story.append(Paragraph(f"Cpk : {metrics['cpk']:.2f}", styles["BodyText"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Conclusion", styles["Heading2"]))

    cpk = metrics["cpk"]
    if cpk < 1:
        conclusion = "Le processus n'est pas conforme aux exigences qualité. Des actions immédiates sont nécessaires."
    elif cpk < 1.33:
        conclusion = "Le processus est limite. Une amélioration est nécessaire."
    else:
        conclusion = "Le processus est globalement maîtrisé."

    story.append(Paragraph(conclusion, styles["BodyText"]))
    doc = SimpleDocTemplate(PDF_PATH)
    doc.build(story)
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


def render_header() -> None:
    h1, h2 = st.columns([1, 5])

    with h1:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=135)

    with h2:
        st.markdown(
            """
            <h1 style="margin:0; font-size:44px; font-weight:900;">SpecSense AI</h1>
            <p style="margin:6px 0 0 0; color:#94a3b8; font-size:18px;">
                Plateforme intelligente de qualité industrielle
            </p>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")


def render_global_kpis(metrics: dict) -> None:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Moyenne", f"{metrics['mean_val']:.4f}")
    k2.metric("Écart-type", f"{metrics['std_val']:.6f}")
    k3.metric("Cp", f"{metrics['cp']:.2f}")
    k4.metric("Cpk", f"{metrics['cpk']:.2f}", "Non capable ⚠️" if metrics["cpk"] < 1.33 else "Capable ✅")
    process_status(metrics["cpk"])
    st.markdown("---")


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

        with col2:
            machine = st.text_input("Machine", value="M1")
            usl = st.number_input("USL", value=12.1000, format="%.4f")
            lsl = st.number_input("LSL", value=11.9000, format="%.4f")

        with col3:
            mesure_1 = st.number_input("Mesure 1", format="%.4f")
            mesure_2 = st.number_input("Mesure 2", format="%.4f")
            mesure_3 = st.number_input("Mesure 3", format="%.4f")

        submitted = st.form_submit_button("Enregistrer")

    if submitted:
        part_id_final = f"{data_type}_{part_id}"

        new_rows = pd.DataFrame([
            {
                "Date_Time": datetime.now(),
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

    tab_control, tab_rules, tab_distribution, tab_capability, tab_machine, tab_ai = st.tabs(
        ["Carte de contrôle", "Règles SPC", "Distribution", "Capabilité", "Machine / Opérateur", "Interprétation IA"]
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

    with tab_capability:
        st.markdown("### 🎯 Capabilité SPC")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("USL", f"{usl:.4f}")
        c2.metric("LSL", f"{lsl:.4f}")
        c3.metric("Cp", f"{metrics['cp']:.2f}")
        c4.metric("Cpk", f"{metrics['cpk']:.2f}")
        process_status(metrics["cpk"])

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
        pdf_path = generate_pdf_report(metrics)
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
    st.caption(f"{APP_NAME} {APP_VERSION} | Qualité 4.0 | Inspiré IATF 16949")


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
        st.stop()
    if df.empty and "manual_data" not in st.session_state:
        st.warning("⚠️ Google Sheet vide — commencez par saisir des données.")
        page_saisie_mesures(df)
        st.stop()

    if "manual_data" in st.session_state:
        df = pd.concat([df, st.session_state["manual_data"]], ignore_index=True)

    metrics = prepare_data(df)

       page = render_sidebar(metrics)
    render_header()
    render_global_kpis(metrics)
       st.write("PAGE:", page)
    # ROUTING
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
