import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="SpecSense AI",
    page_icon="🎯",
    layout="wide"
)

# =========================
# STYLE
# =========================
st.markdown("""
<style>
.stApp {
    background: #0b1117;
    color: white;
}

[data-testid="stSidebar"] {
    background: #1f2330;
}

.big-title {
    font-size: 42px;
    font-weight: 800;
    color: white;
    margin-bottom: 10px;
}

.sub-title {
    color: #aab2c0;
    font-size: 16px;
}

.card {
    background: #151c24;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #293241;
    margin-bottom: 18px;
}

.good {
    color: #22c55e;
    font-weight: bold;
}

.warn {
    color: #facc15;
    font-weight: bold;
}

.bad {
    color: #ef4444;
    font-weight: bold;
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
    df["Measurement"] = pd.to_numeric(df["Measurement"], errors="coerce")
    df["USL"] = pd.to_numeric(df["USL"], errors="coerce")
    df["LSL"] = pd.to_numeric(df["LSL"], errors="coerce")
    df["Severity"] = pd.to_numeric(df["Severity"], errors="coerce")
    df["Occurrence"] = pd.to_numeric(df["Occurrence"], errors="coerce")
    df["Detection"] = pd.to_numeric(df["Detection"], errors="coerce")
    return df.dropna(subset=["Measurement"])

try:
    df = load_data()
except Exception as e:
    st.error("🚨 Ma 9drnach n9raw Google Sheet. Check lien dyal CSV.")
    st.stop()

# =========================
# BASIC DATA
# =========================
msa_data = df[df["Part_ID"].astype(str).str.contains("MSA", na=False)] if "Part_ID" in df.columns else pd.DataFrame()
spc_data = df[df["Part_ID"].astype(str).str.contains("SPC", na=False)] if "Part_ID" in df.columns else df

total = len(df)
msa_count = len(msa_data)
spc_count = len(spc_data)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## 📊 Live KPIs")
    st.markdown("---")
    st.metric("📦 Total Mesures", total)
    st.metric("📏 MSA Points", msa_count)
    st.metric("📈 SPC Points", spc_count)
    st.markdown("---")
    st.markdown(f"🕐 Last Update: {datetime.now().strftime('%H:%M:%S')}")
    st.markdown("---")
    st.caption("© 2026 SpecSense AI")

# =========================
# HEADER
# =========================
st.markdown('<div class="big-title">🎯 SpecSense AI - Quality 4.0 Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Smart Quality Dashboard | SPC • Cpk • MSA • Pareto • FMEA • AI Coach</div>', unsafe_allow_html=True)
st.markdown("---")

# =========================
# TOP KPI CARDS
# =========================
k1, k2, k3, k4 = st.columns(4)

mean_val = df["Measurement"].mean()
std_val = df["Measurement"].std()
usl = df["USL"].iloc[0]
lsl = df["LSL"].iloc[0]

cp = (usl - lsl) / (6 * std_val) if std_val > 0 else 0
cpk = min((usl - mean_val) / (3 * std_val), (mean_val - lsl) / (3 * std_val)) if std_val > 0 else 0

with k1:
    st.metric("Mean", f"{mean_val:.4f}")
with k2:
    st.metric("Std Dev", f"{std_val:.6f}")
with k3:
    st.metric("Cp", f"{cp:.2f}")
with k4:
    st.metric("Cpk", f"{cpk:.2f}", "OK ✅" if cpk >= 1.33 else "Risk ⚠️")

st.markdown("---")

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📏 MSA Gage R&R",
    "📊 SPC Control",
    "📈 Capability Cpk",
    "📋 Pareto Defects",
    "🎯 FMEA RPN",
    "🤖 AI Coach"
])

# =========================
# TAB 1: MSA
# =========================
with tab1:
    st.subheader("📏 MSA Type 1")

    st.write(f"MSA Data: {len(msa_data)} mesures")

    if len(msa_data) > 0:
        mean_msa = msa_data["Measurement"].mean()
        std_msa = msa_data["Measurement"].std()

        ref = (usl + lsl) / 2
        tolerance = usl - lsl

        cg = (0.2 * tolerance) / (6 * std_msa) if std_msa > 0 else 0
        cgk = (0.1 * tolerance - abs(mean_msa - ref)) / (3 * std_msa) if std_msa > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Reference", f"{ref:.4f}")
        c2.metric("Tolerance", f"{tolerance:.4f}")
        c3.metric("Cg", f"{cg:.2f}", "Pass ✅" if cg >= 1.33 else "Fail ❌")
        c4.metric("Cgk", f"{cgk:.2f}", "Pass ✅" if cgk >= 1.33 else "Fail ❌")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=msa_data["Measurement"],
            x=list(range(1, len(msa_data) + 1)),
            mode="lines+markers",
            name="MSA Measurements"
        ))
        fig.add_hline(y=mean_msa, line_dash="dash", annotation_text="Mean")
        fig.add_hline(y=ref, line_dash="dot", annotation_text="Reference")
        fig.update_layout(
            title="MSA Run Chart",
            template="plotly_dark",
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(msa_data, use_container_width=True)
    else:
        st.warning("Ma kaynach data MSA. Khass Part_ID fih MSA.")

# =========================
# TAB 2: SPC
# =========================
with tab2:
    st.subheader("📊 SPC Control Chart")

    if len(spc_data) > 0:
        mean_spc = spc_data["Measurement"].mean()
        std_spc = spc_data["Measurement"].std()

        ucl = mean_spc + 3 * std_spc
        lcl = mean_spc - 3 * std_spc

        c1, c2, c3 = st.columns(3)
        c1.metric("CL", f"{mean_spc:.4f}")
        c2.metric("UCL", f"{ucl:.4f}")
        c3.metric("LCL", f"{lcl:.4f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(spc_data) + 1)),
            y=spc_data["Measurement"],
            mode="lines+markers",
            name="SPC Data"
        ))
        fig.add_hline(y=mean_spc, line_dash="dash", annotation_text="CL")
        fig.add_hline(y=ucl, line_dash="dash", annotation_text="UCL")
        fig.add_hline(y=lcl, line_dash="dash", annotation_text="LCL")
        fig.add_hline(y=usl, line_dash="dot", annotation_text="USL")
        fig.add_hline(y=lsl, line_dash="dot", annotation_text="LSL")

        fig.update_layout(
            title="SPC Individual Control Chart",
            template="plotly_dark",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)

        out_spec = spc_data[
            (spc_data["Measurement"] > usl) |
            (spc_data["Measurement"] < lsl)
        ]

        if len(out_spec) > 0:
            st.error(f"⚠️ Kaynin {len(out_spec)} points kharjin men specification.")
            st.dataframe(out_spec, use_container_width=True)
        else:
            st.success("✅ Kolchi dakhel specification.")
    else:
        st.warning("Ma kaynach data SPC.")

# =========================
# TAB 3: CAPABILITY
# =========================
with tab3:
    st.subheader("📈 Process Capability")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("USL", f"{usl:.4f}")
    c2.metric("LSL", f"{lsl:.4f}")
    c3.metric("Cp", f"{cp:.2f}")
    c4.metric("Cpk", f"{cpk:.2f}", "Capable ✅" if cpk >= 1.33 else "Not Capable ⚠️")

    fig = px.histogram(
        df,
        x="Measurement",
        nbins=20,
        title="Capability Histogram",
        template="plotly_dark"
    )
    fig.add_vline(x=usl, line_dash="dash", annotation_text="USL")
    fig.add_vline(x=lsl, line_dash="dash", annotation_text="LSL")
    fig.add_vline(x=mean_val, line_dash="dot", annotation_text="Mean")
    fig.update_layout(height=450)

    st.plotly_chart(fig, use_container_width=True)

    if cpk >= 1.67:
        st.success("🟢 Excellent process capability.")
    elif cpk >= 1.33:
        st.success("✅ Process capable.")
    elif cpk >= 1.00:
        st.warning("🟡 Process borderline. Khasso improvement.")
    else:
        st.error("🔴 Process not capable.")

# =========================
# TAB 4: PARETO
# =========================
with tab4:
    st.subheader("📋 Pareto Defects")

    defects = df[df["Defect_Type"].astype(str).str.upper() != "OK"]

    if len(defects) > 0:
        pareto = defects["Defect_Type"].value_counts().reset_index()
        pareto.columns = ["Defect_Type", "Count"]
        pareto["Cumulative"] = pareto["Count"].cumsum() / pareto["Count"].sum() * 100

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=pareto["Defect_Type"],
            y=pareto["Count"],
            name="Defects"
        ))
        fig.add_trace(go.Scatter(
            x=pareto["Defect_Type"],
            y=pareto["Cumulative"],
            name="Cumulative %",
            yaxis="y2",
            mode="lines+markers"
        ))
        fig.update_layout(
            title="Pareto Chart",
            template="plotly_dark",
            height=450,
            yaxis=dict(title="Count"),
            yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 110])
        )

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pareto, use_container_width=True)
    else:
        st.success("✅ Ma kaynach defects. Kolchi OK.")

# =========================
# TAB 5: FMEA
# =========================
with tab5:
    st.subheader("🎯 FMEA RPN")

    if all(col in df.columns for col in ["Severity", "Occurrence", "Detection"]):
        fmea = df.copy()
        fmea["RPN"] = fmea["Severity"] * fmea["Occurrence"] * fmea["Detection"]
        fmea = fmea.sort_values("RPN", ascending=False)

        c1, c2, c3 = st.columns(3)
        c1.metric("Max RPN", int(fmea["RPN"].max()))
        c2.metric("Avg RPN", f"{fmea['RPN'].mean():.1f}")
        c3.metric("High Risks", len(fmea[fmea["RPN"] >= 100]))

        fig = px.bar(
            fmea.head(10),
            x="Part_ID",
            y="RPN",
            color="Defect_Type",
            title="Top RPN Risks",
            template="plotly_dark"
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            fmea[["Date_Time", "Part_ID", "Machine", "Defect_Type", "Severity", "Occurrence", "Detection", "RPN"]],
            use_container_width=True
        )
    else:
        st.warning("Columns Severity / Occurrence / Detection khasin.")

# =========================
# TAB 6: AI COACH WITH OPENAI
# =========================
with tab6:
    st.subheader("🤖 AI Quality Coach")

    from openai import OpenAI

    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    defects_count = len(df[df["Defect_Type"].astype(str).str.upper() != "OK"])
    top_defect = "None"

    if defects_count > 0:
        top_defect = df[df["Defect_Type"].astype(str).str.upper() != "OK"]["Defect_Type"].value_counts().idxmax()

    out_spec = df[
        (df["Measurement"] > usl) |
        (df["Measurement"] < lsl)
    ]

    summary = f"""
    Quality data summary:
    Mean = {mean_val:.4f}
    Std Dev = {std_val:.6f}
    USL = {usl}
    LSL = {lsl}
    Cp = {cp:.2f}
    Cpk = {cpk:.2f}
    Defects count = {defects_count}
    Top defect = {top_defect}
    Out of spec points = {len(out_spec)}
    """

    user_question = st.text_area(
        "💬 Sowel AI Coach:",
        placeholder="مثلا: علاش Cpk ضعيف؟ شنو action plan؟ كيفاش نعالج Rayure؟"
    )

    if st.button("🤖 Analyse with AI"):
        if user_question.strip() == "":
            st.warning("كتب شي سؤال الأول.")
        else:
            with st.spinner("AI kayحلل situation..."):
                response = client.responses.create(
                    model="gpt-4.1-mini",
                    input=f"""
                    You are a senior automotive quality engineer.
                    Answer in Moroccan Darija mixed with simple French/English.

                    Use this quality data:
                    {summary}

                    User question:
                    {user_question}

                    Give:
                    1. Interpretation
                    2. Root cause possibilities
                    3. Immediate containment actions
                    4. Corrective actions
                    5. Priority level
                    """
                )

                st.markdown(response.output_text)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("SpecSense AI V1.0 | Quality 4.0 | IATF 16949:2016 Inspired")
