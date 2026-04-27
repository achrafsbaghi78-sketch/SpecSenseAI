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
    initial_sidebar_state="collapsed"
)

# =========================
# STYLE
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071016 0%, #0f172a 100%);
    color: white;
}

[data-testid="stSidebar"] {
    background: #111827;
}

.big-title {
    font-size: 46px;
    font-weight: 900;
    color: white;
    margin-bottom: 12px;
    line-height: 1.1;
}

.sub-title {
    color: #cbd5e1;
    font-size: 20px;
    margin-bottom: 24px;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #0b1a2a, #0f2238);
    padding: 24px;
    border-radius: 22px;
    border: 1px solid rgba(59,130,246,0.2);
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    transition: 0.3s;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-5px);
    border: 1px solid #3b82f6;
}

div[data-testid="stMetricValue"] {
    font-size: 38px;
    font-weight: 900;
}

div[data-testid="stMetricDelta"] {
    font-size: 16px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    overflow-x: auto;
}

.stTabs [data-baseweb="tab"] {
    font-size: 18px;
    font-weight: 700;
    padding: 14px 18px;
    white-space: nowrap;
}

h1, h2, h3 {
    font-weight: 900;
}

h2 {
    font-size: 32px !important;
}

h3 {
    font-size: 26px !important;
}

.stDataFrame {
    font-size: 16px;
}

button {
    border-radius: 16px !important;
    font-size: 17px !important;
    font-weight: 700 !important;
}

@media (max-width: 768px) {
    .big-title {
        font-size: 34px;
    }

    .sub-title {
        font-size: 16px;
    }

    div[data-testid="stMetric"] {
        padding: 18px;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 15px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 30px;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 15px;
        padding: 10px 12px;
    }

    h2 {
        font-size: 26px !important;
    }

    h3 {
        font-size: 22px !important;
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

    for col in ["Measurement", "USL", "LSL", "Severity", "Occurrence", "Detection"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["Measurement"])

try:
    df = load_data()
except Exception:
    st.error("🚨 Impossible de lire Google Sheet. Vérifiez le lien CSV.")
    st.stop()

# =========================
# DATA
# =========================
msa_data = df[df["Part_ID"].astype(str).str.contains("MSA", na=False)] if "Part_ID" in df.columns else pd.DataFrame()
spc_data = df[df["Part_ID"].astype(str).str.contains("SPC", na=False)] if "Part_ID" in df.columns else df

total = len(df)
msa_count = len(msa_data)
spc_count = len(spc_data)

mean_val = df["Measurement"].mean()
std_val = df["Measurement"].std()
usl = df["USL"].iloc[0]
lsl = df["LSL"].iloc[0]

cp = (usl - lsl) / (6 * std_val) if std_val > 0 else 0
cpk = min((usl - mean_val) / (3 * std_val), (mean_val - lsl) / (3 * std_val)) if std_val > 0 else 0

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## 📊 Indicateurs en temps réel")
    st.markdown("---")
    st.metric("📦 Nombre total de mesures", total)
    st.metric("📏 Points MSA", msa_count)
    st.metric("📈 Points SPC", spc_count)
    st.markdown("---")
    st.markdown(f"🕐 Dernière mise à jour : {datetime.now().strftime('%H:%M:%S')}")
    st.markdown("---")
    st.caption("© 2026 SpecSense AI")

# =========================
# HEADER
# =========================
st.markdown('<div class="big-title">🎯 SpecSense AI - Suite Qualité 4.0</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Tableau de bord qualité intelligent • SPC • Cpk • MSA • Pareto • AMDEC • IA</div>', unsafe_allow_html=True)
st.markdown("---")

# =========================
# TOP KPIs
# =========================
k1, k2 = st.columns(2)
k3, k4 = st.columns(2)

with k1:
    st.metric("Moyenne", f"{mean_val:.4f}")
with k2:
    st.metric("Écart-type", f"{std_val:.6f}")
with k3:
    st.metric("Cp", f"{cp:.2f}")
with k4:
    st.metric("Cpk", f"{cpk:.2f}", "Capable ✅" if cpk >= 1.33 else "Non capable ⚠️")

st.markdown("---")

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "MSA",
    "SPC",
    "Capabilité",
    "Pareto",
    "AMDEC",
    "IA"
])

# =========================
# TAB 1: MSA
# =========================
with tab1:
    st.subheader("📏 Analyse MSA Type 1")
    st.write(f"Données MSA : {len(msa_data)} mesures")

    if len(msa_data) > 0:
        mean_msa = msa_data["Measurement"].mean()
        std_msa = msa_data["Measurement"].std()

        ref = (usl + lsl) / 2
        tolerance = usl - lsl

        cg = (0.2 * tolerance) / (6 * std_msa) if std_msa > 0 else 0
        cgk = (0.1 * tolerance - abs(mean_msa - ref)) / (3 * std_msa) if std_msa > 0 else 0

        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)

        c1.metric("Référence", f"{ref:.4f}")
        c2.metric("Tolérance", f"{tolerance:.4f}")
        c3.metric("Cg", f"{cg:.2f}", "Accepté ✅" if cg >= 1.33 else "Refusé ❌")
        c4.metric("Cgk", f"{cgk:.2f}", "Accepté ✅" if cgk >= 1.33 else "Refusé ❌")

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
            title="Carte MSA",
            template="plotly_dark",
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(msa_data, use_container_width=True)
    else:
        st.warning("Aucune donnée MSA disponible.")

# =========================
# TAB 2: SPC
# =========================
with tab2:
    st.subheader("📊 Carte de contrôle SPC")

    if len(spc_data) > 0:
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
            name="Données SPC"
        ))

        fig.add_hline(y=mean_spc, line_dash="dash", annotation_text="CL")
        fig.add_hline(y=ucl, line_dash="dash", annotation_text="UCL")
        fig.add_hline(y=lcl, line_dash="dash", annotation_text="LCL")
        fig.add_hline(y=usl, line_dash="dot", annotation_text="USL")
        fig.add_hline(y=lsl, line_dash="dot", annotation_text="LSL")

        fig.update_layout(
            title="Carte de contrôle individuelle",
            template="plotly_dark",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)

        out_spec = spc_data[
            (spc_data["Measurement"] > usl) |
            (spc_data["Measurement"] < lsl)
        ]

        if len(out_spec) > 0:
            st.error(f"⚠️ {len(out_spec)} points hors spécifications.")
            st.dataframe(out_spec, use_container_width=True)
        else:
            st.success("✅ Tous les points sont conformes.")
    else:
        st.warning("Aucune donnée SPC disponible.")

# =========================
# TAB 3: CAPABILITÉ
# =========================
with tab3:
    st.subheader("📈 Capabilité du processus")

    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)

    c1.metric("USL", f"{usl:.4f}")
    c2.metric("LSL", f"{lsl:.4f}")
    c3.metric("Cp", f"{cp:.2f}")
    c4.metric("Cpk", f"{cpk:.2f}", "Capable ✅" if cpk >= 1.33 else "Non capable ⚠️")

    fig = px.histogram(
        df,
        x="Measurement",
        nbins=20,
        title="Histogramme de capabilité",
        template="plotly_dark"
    )

    fig.add_vline(x=usl, line_dash="dash", annotation_text="USL")
    fig.add_vline(x=lsl, line_dash="dash", annotation_text="LSL")
    fig.add_vline(x=mean_val, line_dash="dot", annotation_text="Moyenne")
    fig.update_layout(height=450)

    st.plotly_chart(fig, use_container_width=True)

    if cpk >= 1.67:
        st.success("🟢 Capabilité excellente.")
    elif cpk >= 1.33:
        st.success("✅ Processus capable.")
    elif cpk >= 1.00:
        st.warning("🟡 Processus limite. Amélioration nécessaire.")
    else:
        st.error("🔴 Processus non capable.")

# =========================
# TAB 4: PARETO
# =========================
with tab4:
    st.subheader("📋 Analyse Pareto des défauts")

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
            name="Cumul %",
            yaxis="y2",
            mode="lines+markers"
        ))

        fig.update_layout(
            title="Diagramme Pareto",
            template="plotly_dark",
            height=450,
            yaxis=dict(title="Nombre"),
            yaxis2=dict(title="Cumul %", overlaying="y", side="right", range=[0, 110])
        )

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pareto, use_container_width=True)
    else:
        st.success("✅ Aucun défaut détecté.")

# =========================
# TAB 5: AMDEC
# =========================
with tab5:
    st.subheader("🎯 Analyse AMDEC automatique")

    if all(col in df.columns for col in ["Severity", "Occurrence", "Detection"]):
        fmea = df.copy()
        fmea["RPN"] = fmea["Severity"] * fmea["Occurrence"] * fmea["Detection"]
        fmea = fmea.sort_values(by="RPN", ascending=False)

        def get_priority(rpn):
            if rpn >= 150:
                return "🔴 Critique"
            elif rpn >= 100:
                return "🟡 Élevé"
            else:
                return "🟢 Moyen"

        fmea["Priorité"] = fmea["RPN"].apply(get_priority)

        c1, c2, c3 = st.columns(3)
        c1.metric("RPN maximum", int(fmea["RPN"].max()))
        c2.metric("RPN moyen", f"{fmea['RPN'].mean():.1f}")
        c3.metric("Risques critiques", len(fmea[fmea["RPN"] >= 150]))

        st.markdown("---")

        table_fmea = fmea[[
            "Part_ID",
            "Defect_Type",
            "Severity",
            "Occurrence",
            "Detection",
            "RPN",
            "Priorité"
        ]].rename(columns={
            "Part_ID": "Référence pièce",
            "Defect_Type": "Type de défaut",
            "Severity": "Gravité",
            "Occurrence": "Occurrence",
            "Detection": "Détection"
        })

        st.dataframe(table_fmea, use_container_width=True)

        top_risk = fmea.iloc[0]

        st.markdown("## 🤖 Actions recommandées par l’IA")
        st.write(f"Risque principal : **{top_risk['Defect_Type']}**")
        st.write(f"RPN : **{top_risk['RPN']}**")

        if top_risk["RPN"] >= 150:
            st.error("🔴 Risque critique - Action immédiate requise")
            st.markdown("""
- Arrêter la production si nécessaire
- Isoler les pièces non conformes
- Vérifier l’état de la machine / du montage
- Réaliser une analyse des causes racines avec les 5 Pourquoi
- Lancer un plan d’actions correctives
""")
        elif top_risk["RPN"] >= 100:
            st.warning("🟡 Risque élevé - Amélioration nécessaire")
            st.markdown("""
- Augmenter la fréquence de contrôle
- Former les opérateurs
- Renforcer la maîtrise du processus
- Réduire la variation
""")
        else:
            st.success("🟢 Risque acceptable")
            st.markdown("""
- Continuer la surveillance
- Maintenir le processus standard
""")
    else:
        st.warning("Colonnes manquantes : Gravité / Occurrence / Détection.")
st.markdown("### 📘 Explication des colonnes AMDEC")

st.info("""
**Gravité** : niveau d’impact du défaut sur le client, le produit ou la sécurité.  
Plus la gravité est élevée, plus le défaut est critique.

**Occurrence** : fréquence d’apparition du défaut.  
Plus l’occurrence est élevée, plus le défaut se répète souvent.

**Détection** : capacité à détecter le défaut avant livraison au client.  
Plus la détection est élevée, plus le défaut est difficile à détecter.

**RPN** : niveau de risque global.  
RPN = Gravité × Occurrence × Détection.

**Priorité** : classement automatique du risque :
- 🔴 Critique : action immédiate requise
- 🟡 Élevé : amélioration nécessaire
- 🟢 Moyen : surveillance continue
""")
# =========================
# TAB 6: IA HUGGING FACE
# =========================
with tab6:
    st.subheader("🤖 Assistant Qualité IA")

    from huggingface_hub import InferenceClient

    try:
        client = InferenceClient(token=st.secrets["HF_TOKEN"])
    except Exception:
        st.error("🚨 HF_TOKEN manquant dans Streamlit Secrets.")
        st.stop()

    if "hf_messages" not in st.session_state:
        st.session_state.hf_messages = []

    if st.button("🧹 Réinitialiser le chat"):
        st.session_state.hf_messages = []

    for msg in st.session_state.hf_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("💬 Posez votre question qualité...")

    if prompt:
        st.session_state.hf_messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        context = f"""
Tu es un ingénieur qualité automobile senior.

Réponds uniquement en français avec un style simple, clair et professionnel.

Données qualité :
Moyenne = {mean_val:.4f}
Écart-type = {std_val:.6f}
Cp = {cp:.2f}
Cpk = {cpk:.2f}
USL = {usl}
LSL = {lsl}

Ta réponse doit contenir :
- Interprétation
- Causes possibles
- Actions immédiates
- Actions correctives
- Priorité
"""

        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": prompt}
        ]

        with st.chat_message("assistant"):
            with st.spinner("L’IA analyse la situation..."):
                try:
                    response = client.chat.completions.create(
                        model="meta-llama/Meta-Llama-3-8B-Instruct",
                        messages=messages,
                        max_tokens=500,
                        temperature=0.3
                    )

                    reply = response.choices[0].message.content
                    st.markdown(reply)

                    st.session_state.hf_messages.append({
                        "role": "assistant",
                        "content": reply
                    })

                except Exception as e:
                    st.error(f"❌ Erreur IA : {e}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("SpecSense AI V1.0 | Qualité 4.0 | Inspiré IATF 16949")
