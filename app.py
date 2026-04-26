import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# ============================================
# CONFIG
# ============================================
st.set_page_config(
    page_title="SpecSense AI", 
    page_icon="🎯", 
    layout="wide"
)

# ============================================
# GOOGLE SHEET - BDL HNA LINK DYALK
# ============================================
G_SHEET_URL = "https://docs.google.com/spreadsheets/d/1hDJ1A1BSYNoeSMCB0E07aP0vSv1HbqVjmaR3xzYbi74/export?format=csv&gid=0"

# ============================================
# LOAD DATA
# ============================================
@st.cache_data(ttl=60)
def load_data():
    df = pd.read_csv(G_SHEET_URL)
    return df

try:
    df = load_data()
except:
    st.error("🚨 Error: Ma 9drnach n9raw Google Sheet. Check l link.")
    st.stop()

# ============================================
# SIDEBAR KPIs
# ============================================
with st.sidebar:
    st.markdown("## 📊 Live KPIs")
    st.markdown("---")
    st.metric(label="📦 Total Mesures", value=len(df))
    st.metric(label="📏 MSA Points", value=len(df[df['Part_ID'].str.contains('MSA', na=False)]))
    st.metric(label="📈 SPC Points", value=len(df[df['Part_ID'].str.contains('SPC', na=False)]))
    st.markdown("---")
    st.markdown(f"🕐 **Last Update** \n{datetime.now().strftime('%H:%M:%S')}")
    st.markdown("---")
    st.caption("© 2026 SpecSense AI")

# ============================================
# HEADER
# ============================================
st.title("🎯 SpecSense AI - Quality 4.0 Suite")
st.markdown("---")

# ============================================
# TABS ORIGINAL
# ============================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📏 MSA Gage R&R", 
    "📊 SPC X̄-R", 
    "📈 Capability Cpk", 
    "📋 Pareto Defects", 
    "🎯 FMEA RPN",
    "🤖 AI Coach"
])

# ============================================
# TAB 1: MSA
# ============================================
with tab1:
    st.subheader("📏 MSA Type 1 + Gage R&R Study")
    
    msa_data = df[df['Part_ID'].str.contains('MSA', na=False)]
    
    if len(msa_data) > 0:
        # HNA CODE MSA DYALK LI KAN KHDDAM
        st.write("MSA Data:", len(msa_data), "mesures")
        
        # Example: Cg Cgk
        if 'MSA_Ref' in df.columns and 'Tolerance' in df.columns:
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
    else:
        st.warning("⚠️ Ma kaynch MSA data f Sheet")

# ============================================
# TAB 2: SPC
# ============================================
with tab2:
    st.subheader("📊 SPC X̄-R Control Chart")
    
    spc_data = df[df['Part_ID'].str.contains('SPC', na=False)]
    
    if len(spc_data) > 0:
        st.write("SPC Data:", len(spc_data), "points")
        # HNA ZID CODE SPC DYALK
        st.line_chart(spc_data['Measurement'])
    else:
        st.info("📊 SPC - Mazal ma kaynch data SPC f Sheet")

# ============================================
# TAB 3: CPK
# ============================================
with tab3:
    st.subheader("📈 Process Capability - Cpk/Ppk")
    
    if 'USL' in df.columns and 'LSL' in df.columns:
        usl = df['USL'].iloc[0]
        lsl = df['LSL'].iloc[0]
        mean = df['Measurement'].mean()
        std = df['Measurement'].std()
        
        cp = (usl - lsl) / (6 * std) if std > 0 else 0
        cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std)) if std > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cp", f"{cp:.2f}")
        with col2:
            st.metric("Cpk", f"{cpk:.2f}", delta="Not Capable ⚠️" if cpk < 1.33 else "Capable ✅")
        with col3:
            sigma = (cpk * 3) if cpk > 0 else 0
            st.metric("Sigma Level", f"{sigma:.1f}σ")
    else:
        st.warning("⚠️ Khass USL w LSL f Google Sheet")

# ============================================
# TAB 4: PARETO
# ============================================
with tab4:
    st.subheader("📋 Pareto Analysis - Defects")
    st.info("🚧 Pareto - Mazal khassna n3mroha")

# ============================================
# TAB 5: FMEA
# ============================================
with tab5:
    st.subheader("🎯 FMEA Risk Priority Number")
    st.info("🚧 FMEA - Mazal khassna n3mroha")

# ============================================
# TAB 6: AI COACH
# ============================================
with tab6:
    st.subheader("🤖 SpecSense AI Coach - Decision Maker")
    st.caption("3tini ay KPI w ngolik mochkil + 7al + Action Plan IATF 16949")
    
    col1, col2 = st.columns([1,2])
    
    with col1:
       kpi_type = st.selectbox("Chno KPI?", ["Cpk", "Cp", "Ppk", "Cg", "Cgk", "RPN"])
