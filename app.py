import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

st.set_page_config(page_title="SpecSense AI", page_icon="🎯", layout="wide")

# GOOGLE SHEET DYALK
G_SHEET_URL = "https://docs.google.com/spreadsheets/d/1Xy4tgkGs1OXOTh-OMAsR7YsfkUPxttF7qalhDdhHa90/export?format=csv&gid=0"

@st.cache_data(ttl=60)
def load_data():
    df = pd.read_csv(G_SHEET_URL)
    return df

try:
    df = load_data()
except:
    st.error("🚨 Error: Ma 9drnach n9raw Google Sheet")
    st.stop()

# SIDEBAR
with st.sidebar:
    st.markdown("## 📊 Live KPIs")
    st.markdown("---")
    st.metric("📦 Total Mesures", len(df))
    if 'Part_ID' in df.columns:
        st.metric("📏 MSA Points", len(df[df['Part_ID'].str.contains('MSA', na=False)]))
        st.metric("📈 SPC Points", len(df[df['Part_ID'].str.contains('SPC', na=False)]))
    st.markdown("---")
    st.markdown(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
    st.caption("© 2026 SpecSense AI")

# HEADER
st.title("🎯 SpecSense AI - Quality 4.0 Suite")
st.markdown("---")

# HADI HIA LINE LI KAT3RF TAB1 - KHASSA TKOUN HNA 9BL MN WITH TAB1
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📏 MSA Gage R&R", 
    "📊 SPC X̄-R", 
    "📈 Capability Cpk", 
    "📋 Pareto Defects", 
    "🎯 FMEA RPN",
    "🤖 AI Coach"
])

# TAB 1: MSA
with tab1:
    st.subheader("📏 MSA Type 1")
    if 'Part_ID' in df.columns:
        msa_data = df[df['Part_ID'].str.contains('MSA', na=False)]
        st.write(f"MSA Data: {len(msa_data)} mesures")
        
        if len(msa_data) > 0 and all(col in df.columns for col in ['MSA_Ref', 'Tolerance']):
            ref = df['MSA_Ref'].iloc[0]
            tol = df['Tolerance'].iloc[0]
            mean_msa = msa_data['Measurement'].mean()
            std_msa = msa_data['Measurement'].std()
            
            cg = (0.2 * tol) / (6 * std_msa) if std_msa > 0 else 0
            cgk = (0.1 * tol - abs(mean_msa - ref)) / (3 * std_msa) if std_msa > 0 else 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Cg", f"{cg:.2f}", delta="Pass ✅" if cg >= 1.33 else "Fail ❌")
            with col2:
                st.metric("Cgk", f"{cgk:.2f}", delta="Pass ✅" if cgk >= 1.33 else "Fail ❌")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=msa_data['Measurement'], mode='lines+markers', name='Mesures'))
            fig.add_hline(y=ref, line_dash="dash", line_color="green", annotation_text="Ref")
            fig.update_layout(title="MSA Run Chart", template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

# TAB 2
with tab2:
    st.subheader("📊 SPC X̄-R")
    st.write("SPC - Under Construction")

# TAB 3
with tab3:
    st.subheader("📈 Capability Cpk")
    if all(col in df.columns for col in ['USL', 'LSL', 'Measurement']):
        usl = df['USL'].iloc[0]
        lsl = df['LSL'].iloc[0]
        mean = df['Measurement'].mean()
        std = df['Measurement'].std()
        cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std)) if std > 0 else 0
        st.metric("Cpk", f"{cpk:.2f}", delta="Capable ✅" if cpk >= 1.33 else "Not Capable ⚠️")

# TAB 4
with tab4:
    st.subheader("📋 Pareto Defects")
    st.info("🚧 Mazal")

# TAB 5
with tab5:
    st.subheader("🎯 FMEA RPN")
    st.info("🚧 Mazal")

# TAB 6
with tab6:
    st.subheader("🤖 AI Coach")
    st.write("AI Coach - Under Construction")

st.markdown("---")
st.caption("SpecSense AI V1.0 | IATF 16949:2016 Compliant")
