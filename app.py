import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ============================================
# CONFIG + GOOGLE SHEET LINK
# ============================================
st.set_page_config(page_title="SpecSense AI", page_icon="🎯", layout="wide")
# ============================================
# PROFESSIONAL LIGHT THEME - SPECSENSE V2.0
# ============================================
st.set_page_config(
    page_title="SpecSense AI", 
    page_icon="🎯", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Blanc + Bleu Pro
st.markdown("""
<style>
    /* Main App Background */
    .stApp {
        background-color: #F8F9FA;
    }
    
    /* Sidebar - Bleu Foncé Pro */
    [data-testid="stSidebar"] {
        background-color: #1E3A5F;
    }
    
    /* Sidebar Text - Blanc */
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    
    /* Metrics Cards */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #1E3A5F;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1E3A5F !important;
        font-weight: 700 !important;
    }
    
    /* Tabs - Modern */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #FFFFFF;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #E9ECEF;
        border-radius: 8px;
        color: #1E3A5F;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1E3A5F;
        color: white !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #1E3A5F;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 10px 24px;
    }
    
    .stButton>button:hover {
        background-color: #2C5282;
    }
    
    /* Dataframe */
    .dataframe {
        border: none !important;
        border-radius: 8px;
    }
    
    /* Success/Error/Warning Boxes */
    .stAlert {
        border-radius: 8px;
        border-left: 5px solid;
    }
    
</style>
""", unsafe_allow_html=True)
# BDL HNA B LINK DYAL GOOGLE SHEET DYALK
# MOHIM: Khass ykoun f lakher: /export?format=csv&gid=0
G_SHEET_URL = "https://docs.google.com/spreadsheets/d/1Xy4tgkGs1OXOTh-OMAsR7YsfkUPxttF7qalhDdhHa90/export?format=csv&gid=0"

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
        st.info("Vérifier: 1) Link fih /export?format=csv 2) Sheet shared 'Anyone with link'")
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
st.sidebar.metric("Last Update", datetime.now().strftime('%H:%M:%S'))

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
        std = msa_df['Measurement'].std()
        mean = msa_df['Measurement'].mean()
        
        cg = (0.2 * (usl - lsl)) / (6 * std) if std > 0 else 0
        cgk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std)) if std > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Cg", f"{cg:.2f}", "✅ Pass" if cg >= 1.33 else "❌ Fail")
        col2.metric("Cgk", f"{cgk:.2f}", "✅ Pass" if cgk >= 1.33 else "❌ Fail")
        col3.metric("Bias", f"{mean - ref:.4f}")
        
        fig = px.histogram(msa_df, x='Measurement', nbins=20, title="MSA Distribution")
        fig.add_vline(x=ref, line_dash="dash", line_color="green", annotation_text="Reference")
        fig.add_vline(x=usl, line_dash="dash", line_color="red", annotation_text="USL")
        fig.add_vline(x=lsl, line_dash="dash", line_color="red", annotation_text="LSL")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Makaynch MSA data f Sheet. Zid rows fihom Part_ID = 'MSA_001'")

# TAB 2: SPC
with tab2:
    st.subheader("📊 SPC Live - X̄-R Chart")
    spc_df = df[df['Part_ID'].str.contains('SPC', na=False)].sort_values('Date_Time')
    if len(spc_df) >= 5:
        spc_df['Subgroup'] = spc_df.index // 5
        xbar = spc_df.groupby('Subgroup')['Measurement'].mean().reset_index()
        r_chart = spc_df.groupby('Subgroup')['Measurement'].agg(lambda x: x.max() - x.min()).reset_index()
        xbar['Range'] = r_chart['Measurement']
        
        xbar_mean = xbar['Measurement'].mean()
        r_mean = xbar['Range'].mean()
        A2 = 0.577
        D4 = 2.114
        D3 = 0
        
        UCL_X = xbar_mean + A2 * r_mean
        LCL_X = xbar_mean - A2 * r_mean
        UCL_R = D4 * r_mean
        LCL_R = D3 * r_mean
        
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xbar['Subgroup'], y=xbar['Measurement'], mode='lines+markers', name='X̄'))
            fig.add_hline(y=UCL_X, line_dash="dash", line_color="red", annotation_text="UCL")
            fig.add_hline(y=xbar_mean, line_color="green", annotation_text="CL")
            fig.add_hline(y=LCL_X, line_dash="dash", line_color="red", annotation_text="LCL")
            fig.update_layout(title="X̄ Chart", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=xbar['Subgroup'], y=xbar['Range'], mode='lines+markers', name='R'))
            fig2.add_hline(y=UCL_R, line_dash="dash", line_color="red", annotation_text="UCL")
            fig2.add_hline(y=r_mean, line_color="green", annotation_text="CL")
            fig2.update_layout(title="R Chart", height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        violations = xbar[(xbar['Measurement'] > UCL_X) | (xbar['Measurement'] < LCL_X)]
        if not violations.empty:
            st.error(f"🚨 {len(violations)} Out-of-Control Points Detected!")
        else:
            st.success("✅ Process In Control")
    else:
        st.info("Khass 5 points SPC 3la a9al. Zid rows fihom Part_ID = 'SPC_001'")

# TAB 3: CAPABILITY
with tab3:
    st.subheader("📈 Process Capability - Cpk/Ppk")
    spc_df = df[df['Part_ID'].str.contains('SPC', na=False)]
    if not spc_df.empty:
        usl = spc_df['USL'].iloc[0]
        lsl = spc_df['LSL'].iloc[0]
        mean = spc_df['Measurement'].mean()
        std = spc_df['Measurement'].std()
        
        if std > 0:
            cp = (usl - lsl) / (6 * std)
            cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Cp", f"{cp:.2f}", "✅ Capable" if cp >= 1.33 else "⚠️ Not Capable")
            col2.metric("Cpk", f"{cpk:.2f}", "✅ Capable" if cpk >= 1.33 else "⚠️ Not Capable")
            col3.metric("Sigma Level", f"{3*cpk:.1f}σ")
            
            fig = px.histogram(spc_df, x='Measurement', nbins=30, marginal="box", title="Process Distribution vs Specs")
            fig.add_vline(x=usl, line_dash="dash", line_color="red", annotation_text="USL")
            fig.add_vline(x=lsl, line_dash="dash", line_color="red", annotation_text="LSL")
            fig.add_vline(x=mean, line_color="green", annotation_text="Mean")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Std = 0. Vérifier data")
    else:
        st.info("Makaynch SPC data")

# TAB 4: PARETO
with tab4:
    st.subheader("📋 Pareto Analysis - 80/20 Rule")
    if 'Defect_Type' in df.columns and not df['Defect_Type'].isna().all():
        pareto = df['Defect_Type'].value_counts().reset_index()
        pareto.columns = ['Defect', 'Count']
        pareto['Cum%'] = 100 * pareto['Count'].cumsum() / pareto['Count'].sum()
        
        fig = px.bar(pareto, x='Defect', y='Count', title="Pareto Chart - Vital Few")
        fig.add_scatter(x=pareto['Defect'], y=pareto['Cum%'], mode='lines+markers', name='Cum %', yaxis='y2')
        fig.update_layout(yaxis2=dict(overlaying='y', side='right', range=[0,100], title='Cumulative %'))
        st.plotly_chart(fig, use_container_width=True)
        
        vital_few = pareto[pareto['Cum%'] <= 80]
        st.info(f"🎯 Vital Few: {', '.join(vital_few['Defect'].tolist())} = 80% dyal defects")
    else:
        st.info("Makaynch Defect_Type data f Sheet")

# TAB 5: FMEA
with tab5:
    st.subheader("🎯 FMEA Risk Analysis - RPN")
    if all(col in df.columns for col in ['Severity', 'Occurrence', 'Detection', 'Defect_Type']):
        fmea = df[['Defect_Type', 'Severity', 'Occurrence', 'Detection']].dropna()
        fmea['RPN'] = fmea['Severity'] * fmea['Occurrence'] * fmea['Detection']
        fmea = fmea.sort_values('RPN', ascending=False)
        
        def risk_level(rpn):
            if rpn >= 100: return "🔴 High"
            elif rpn >= 50: return "🟠 Medium"
            else: return "🟢 Low"
        
        fmea['Risk'] = fmea['RPN'].apply(risk_level)
        st.dataframe(fmea, use_container_width=True, hide_index=True)
        
        fig = px.bar(fmea.head(10), x='Defect_Type', y='RPN', color='Risk', 
                     color_discrete_map={"🔴 High":"red", "🟠 Medium":"orange", "🟢 Low":"green"},
                     title="Top 10 Risks by RPN")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Khass Severity, Occurrence, Detection, Defect_Type f Sheet")
# ============================================
# TAB 6: AI QUALITY COACH 🤖
# ============================================
tab6 = st.tabs(["🤖 AI Coach"])[0]

with tab6:
    st.subheader("🤖 SpecSense AI Coach - Decision Maker")
    st.caption("3tini ay KPI w ngolik mochkil + 7al + Action Plan IATF 16949")
    
    col1, col2 = st.columns([1,2])
    
    with col1:
        kpi_type = st.selectbox("Chno KPI?", ["Cpk", "Cp", "Ppk", "Cg", "Cgk", "RPN"])
        kpi_value = st.number_input("Dkhl Value", value=1.0, step=0.01, format="%.2f")
        
        if st.button("🚀 Analyse Liya", use_container_width=True):
            st.session_state.analyze = True
    
    with col2:
        if 'analyze' in st.session_state:
            st.markdown("### 📋 Diagnostic + Action Plan")
            
            # CPK LOGIC
            if kpi_type == "Cpk":
                if kpi_value >= 1.67:
                    st.success("✅ **Excellent - World Class**")
                    st.write("**Tafsir:** Process capable bzaf. 6 Sigma level.")
                    st.write("**Action:** 1) Maintain current controls 2) Reduce inspection frequency 3) Use for customer audit")
                elif kpi_value >= 1.33:
                    st.success("✅ **Good - IATF Compliant**")
                    st.write("**Tafsir:** Process capable. Conforme IATF 16949.")
                    st.write("**Action:** 1) Continue SPC monitoring 2) Review if customer requires Cpk>1.67")
                elif kpi_value >= 1.00:
                    st.warning("⚠️ **Marginal - Risk**")
                    st.write("**Mochkil:** Process 3la 7d. Ay shift sghira = Non-conforme.")
                    st.write("**7al IATF 8.5.1:** 1) 100% Sort for next 3 lots 2) 8D Analysis 3) Process improvement Kaizen 4) Re-qualify")
                else:
                    st.error("🚨 **Not Capable - NOK**")
                    st.write("**Mochkil:** Process ma 9adch y produce f tolerance. Defects ghadi ykhrjo.")
                    st.write("**7al Urgent IATF 8.7:** 1) STOP PRODUCTION 2) Containment lot précédent 3) 8D + 5Why 4) Corrective: Tooling/Fixture/Parameter 5) Re-PPAP")
            
            # CG LOGIC - MSA
            elif kpi_type == "Cg":
                if kpi_value >= 1.33:
                    st.success("✅ **MSA Pass - IATF 7.1.5.1**")
                    st.write("**Tafsir:** L'appareil stable w repeatable.")
                    st.write("**Action:** 1) Document MSA report 2) Calibration OK 3) Use for production")
                else:
                    st.error("🚨 **MSA Fail - Non-Compliant**")
                    st.write("**Mochkil:** Variation dyal appareil ktira. Ma t9drch t9iss b ti9a.")
                    st.write("**7al IATF 7.1.5.2:** 1) STOP - Ma tsta3mlch appareil 2) Calibration externe 3) Training opérateur 4) Gage R&R complet 5) Ila ba9i NOK = Change equipment")
            
            # RPN LOGIC - FMEA
            elif kpi_type == "RPN":
                if kpi_value >= 100:
                    st.error("🔴 **High Risk - Action Required IATF 6.1**")
                    st.write("**Mochkil:** Risk kbir. Customer impact possible.")
                    st.write("**7al:** 1) Action prioritaire <30 jours 2) Poka-Yoke 3) Control Plan update 4) Re-calculate RPN after action <50")
                elif kpi_value >= 50:
                    st.warning("🟠 **Medium Risk**")
                    st.write("**Action:** Plan d'amélioration 90 jours. Monitor f SPC.")
                else:
                    st.success("🟢 **Low Risk - Acceptable**")
                    st.write("**Action:** Monitor only. No action required.")
            
            # CP LOGIC
            elif kpi_type == "Cp":
                if kpi_value >= 1.33:
                    st.success("✅ **Process Potential OK**")
                    st.write("**Tafsir:** Dispersion mzyana. Ila centré ghadi ykoun Cpk=CP")
                else:
                    st.error("🚨 **Variation Ktira**")
                    st.write("**Mochkil:** Process ma stable. Khass reduction variation.")
                    st.write("**7al:** 1) DOE study 2) Identify X variables 3) SPC + 6M analysis")
            
            st.info("💡 **Note IATF:** Kol action khassha tkoun documented f Control Plan + FMEA + 8D ila tlb l7al")
st.markdown("---")
st.caption(f"SpecSense AI v1.0 | IATF 16949:2016 Compliant | Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
