import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ============================================
# CONFIG + GOOGLE SHEET LINK
# ============================================
st.set_page_config(page_title="SpecSense AI", page_icon="🎯", layout="wide")

# BDL HNA B LINK DYAL GOOGLE SHEET DYALK
G_SHEET_URL = "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/export?format=csv&gid=0"

# ============================================
# LOAD DATA
# ============================================
@st.cache_data(ttl=10)
def load_data():
    try:
        df = pd.read_csv(G_SHEET_URL)
        df['Date_Time'] = pd.to_datetime(df['Date_Time'])
        return df
    except Exception as e:
        st.error(f"⚠️ Error loading Google Sheet: {e}")
        return pd.DataFrame()

df = load_data()

# ============================================
# HEADER
# ============================================
st.title("🎯 SpecSense AI - Quality 4.0 Suite")
st.caption("MSA Gage R&R + SPC Live + Cpk + FMEA + Pareto | IATF 16949:2016 Compliant")

if df.empty:
    st.warning("⚠️ Sheet khawi wla link ghaalt. Vérifier G_SHEET_URL + Share Settings")
    st.stop()

# ============================================
# SIDEBAR KPIs
# ============================================
st.sidebar.header("📊 KPIs Live")
st.sidebar.metric("Total Mesures", len(df))
st.sidebar.metric("MSA Points", len(df[df['Part_ID'].str.contains('MSA', na=False)]))
st.sidebar.metric("SPC Points", len(df[df['Part_ID'].str.contains('SPC', na=False)]))

# ============================================
# TABS
# ============================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📏 MSA Gage R&R", "📊 SPC X̄-R", "📈 Capability Cpk", "📋 Pareto Defects", "🎯 FMEA RPN"])

# TAB 1: MSA
with tab1:
    st.subheader("📏 MSA Type 1 + Gage R&R Study")
    msa_df = df[df['Part_ID'].str.contains('MSA', na=False)]
    if not msa_df.empty:
        usl = msa_df['USL'].iloc[0]
        lsl = msa_df['LSL'].iloc[0]
        ref = (usl + lsl) / 2
        cg = (0.2 * (usl - lsl)) / (6 * msa_df['Measurement'].std())
        cgk = min((usl - msa_df['Measurement'].mean()) / (3 * msa_df['Measurement'].std()),
                  (msa_df['Measurement'].mean() - lsl) / (3 * msa_df['Measurement'].std()))
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Cg", f"{cg:.2f}", "✅ Pass" if cg >= 1.33 else "❌ Fail")
        col2.metric("Cgk", f"{cgk:.2f}", "✅ Pass" if cgk >= 1.33 else "❌ Fail")
        col3.metric("Bias", f"{msa_df['Measurement'].mean() - ref:.4f}")
        
        fig = px.histogram(msa_df, x='Measurement', nbins=20, title="MSA Distribution")
        fig.add_vline(x=ref, line_dash="dash", line_color="green", annotation_text="Reference")
        fig.add_vline(x=usl, line_dash="dash", line_color="red", annotation_text="USL")
        fig.add_vline(x=lsl, line_dash="dash", line_color="red", annotation_text="LSL")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Makaynch MSA data f Sheet")

# TAB 2: SPC
with tab2:
    st.subheader("📊 SPC Live - X̄-R Chart")
    spc_df = df[df['Part_ID'].str.contains('SPC', na=False)].sort_values('Date_Time')
    if len(spc_df) >= 5:
        spc_df['Subgroup'] = spc_df.index // 5
        xbar = spc_df.groupby('Subgroup')['Measurement'].mean().reset_index()
        r_chart = spc_df.groupby('Subgroup')['Measurement'].agg(lambda x: x.max() - x.min()).reset_index()
        
        xbar_mean = xbar['Measurement'].mean()
        r_mean = r_chart['Measurement'].mean()
        A2 = 0.577 # n=5
        UCL_X = xbar_mean + A2 * r_mean
        LCL_X = xbar_mean - A2 * r_mean
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xbar['Subgroup'], y=xbar['Measurement'], mode='lines+markers', name='X̄'))
        fig.add_hline(y=UCL_X, line_dash="dash", line_color="red", annotation_text="UCL")
        fig.add_hline(y=xbar_mean, line_color="green", annotation_text="CL
