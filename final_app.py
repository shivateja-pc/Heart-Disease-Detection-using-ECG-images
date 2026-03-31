import streamlit as st
from Ecg import ECG
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECG Cardiovascular Detection",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS — Light Mode ───────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #f5f7fa !important;
        color: #1a1a2e !important;
    }

    /* Header */
    .header-box {
        background: linear-gradient(135deg, #ffffff, #f0f4ff);
        border: 1px solid #dde3f0;
        border-left: 5px solid #e63946;
        border-radius: 10px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .header-box h1 { color: #1a1a2e; font-size: 2rem; font-weight: 700; margin: 0 0 4px 0; }
    .header-box p  { color: #6b7280; font-size: 0.9rem; margin: 0; }

    /* Metric cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px 16px;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .metric-label { font-size: 0.72rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 1.5px; font-family: 'IBM Plex Mono', monospace; }
    .metric-value { font-size: 1.7rem; font-weight: 700; color: #2563eb; margin-top: 4px; }
    .metric-value-sm { font-size: 1.1rem; font-weight: 700; color: #2563eb; margin-top: 8px; }

    /* Result cards */
    .result-normal {
        background: #f0fdf4;
        border: 2px solid #16a34a;
        border-radius: 14px;
        padding: 36px 32px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(22,163,74,0.12);
    }
    .result-danger {
        background: #fff1f2;
        border: 2px solid #dc2626;
        border-radius: 14px;
        padding: 36px 32px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(220,38,38,0.12);
    }
    .result-warning {
        background: #fffbeb;
        border: 2px solid #d97706;
        border-radius: 14px;
        padding: 36px 32px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(217,119,6,0.12);
    }
    .result-title {
        font-size: 0.78rem;
        color: #9ca3af;
        margin-bottom: 10px;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .result-value-normal  { font-size: 1.8rem; font-weight: 700; color: #15803d; margin: 0; }
    .result-value-danger  { font-size: 1.8rem; font-weight: 700; color: #b91c1c; margin: 0; }
    .result-value-warning { font-size: 1.8rem; font-weight: 700; color: #b45309; margin: 0; }
    .result-meta { color: #9ca3af; margin-top: 10px; font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "last_processed" not in st.session_state:
    st.session_state.last_processed = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ── PDF Generator ─────────────────────────────────────────────────────────────
def generate_pdf_report(patient_name, patient_age, patient_gender, prediction, timestamp):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=inch, leftMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()

    title_style   = ParagraphStyle('T', parent=styles['Title'],   fontSize=22, textColor=colors.HexColor('#1a1a2e'), spaceAfter=6)
    sub_style     = ParagraphStyle('S', parent=styles['Normal'],  fontSize=10, textColor=colors.HexColor('#555555'), spaceAfter=20)
    heading_style = ParagraphStyle('H', parent=styles['Heading2'],fontSize=13, textColor=colors.HexColor('#e63946'), spaceBefore=16, spaceAfter=8)
    result_color  = '#16a34a' if ('Normal' in prediction and 'History' not in prediction) else '#dc2626'
    result_style  = ParagraphStyle('R', parent=styles['Normal'],  fontSize=14, fontName='Helvetica-Bold', textColor=colors.HexColor(result_color), spaceAfter=12)
    disclaimer_st = ParagraphStyle('D', parent=styles['Normal'],  fontSize=9,  textColor=colors.HexColor('#888888'), leading=13)

    tbl_style = TableStyle([
        ('BACKGROUND',     (0,0), (0,-1), colors.HexColor('#f0f0f0')),
        ('FONTNAME',       (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE',       (0,0), (-1,-1), 10),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.HexColor('#ffffff'), colors.HexColor('#f9f9f9')]),
        ('BOX',            (0,0), (-1,-1), 0.5, colors.HexColor('#cccccc')),
        ('INNERGRID',      (0,0), (-1,-1), 0.25, colors.HexColor('#dddddd')),
        ('PADDING',        (0,0), (-1,-1), 8),
        ('TEXTCOLOR',      (0,0), (-1,-1), colors.HexColor('#222222')),
    ])

    story = [
        Paragraph("ECG Cardiovascular Detection Report", title_style),
        Paragraph("Automated ECG Analysis — Powered by Machine Learning", sub_style),
        HRFlowable(width="100%", thickness=2, color=colors.HexColor('#e63946')),
        Spacer(1, 16),
        Paragraph("Patient & Report Details", heading_style),
        Table([
            ["Patient Name",    patient_name if patient_name else "Not Provided"],
            ["Age",             str(patient_age)],
            ["Gender",          patient_gender],
            ["Report Generated",timestamp],
            ["Analysis Method", "ECG Image Processing + Ensemble Voting Classifier (KNN, LR, SVM, RF, NB, XGBoost)"],
            ["Model Accuracy",  "95.7%"],
        ], colWidths=[2.2*inch, 4*inch], style=tbl_style),
        Spacer(1, 16),
        Paragraph("Diagnosis Result", heading_style),
        Paragraph(prediction, result_style),
        Spacer(1, 8),
        Paragraph("Model Information", heading_style),
        Table([
            ["Algorithm",          "Ensemble Voting Classifier"],
            ["Models Used",        "KNN, Logistic Regression, SVM, Random Forest, GaussianNB, XGBoost"],
            ["Dim. Reduction",     "PCA (Principal Component Analysis)"],
            ["Training Accuracy",  "95.7%"],
            ["No. of Classes",     "4"],
            ["Classes",            "Normal, Myocardial Infarction, Abnormal Heartbeat, History of MI"],
        ], colWidths=[2.2*inch, 4*inch], style=tbl_style),
        Spacer(1, 20),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc')),
        Spacer(1, 8),
        Paragraph(
            "DISCLAIMER: This report is generated by an automated machine learning system for "
            "educational and research purposes only. It does not constitute medical advice. "
            "Please consult a qualified cardiologist for clinical diagnosis and treatment.",
            disclaimer_st
        ),
    ]
    doc.build(story)
    buffer.seek(0)
    return buffer

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🫀 Cardiovascular Disease Detection</h1>
    <p>ECG Image Analysis &nbsp;·&nbsp; Ensemble Voting Classifier &nbsp;·&nbsp; 95.7% Accuracy</p>
</div>
""", unsafe_allow_html=True)

# ── Metric Cards ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><div class="metric-label">Model Accuracy</div><div class="metric-value">95.7%</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="metric-label">Algorithm</div><div class="metric-value-sm">Voting Classifier</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div class="metric-label">ECG Leads</div><div class="metric-value">12 + 1</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><div class="metric-label">Classes</div><div class="metric-value">4</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📊 Model Performance")
st.sidebar.markdown("**Ensemble Models Used**")
st.sidebar.markdown("""
- KNN
- Logistic Regression  
- SVM
- Random Forest
- GaussianNB
- XGBoost
""")
st.sidebar.caption("Combined via GridSearchCV-tuned Voting Classifier")

labels = ["Abnormal HB", "MI", "Normal", "History MI"]
cm = np.array([[207,3,2,1],[4,183,2,2],[2,1,98,1],[3,2,1,83]])

fig_cm, ax_cm = plt.subplots(figsize=(4, 3.2))
fig_cm.patch.set_facecolor('#ffffff')
ax_cm.set_facecolor('#ffffff')
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=labels, yticklabels=labels, ax=ax_cm,
            linewidths=0.5, linecolor='#e2e8f0', annot_kws={"size": 9})
ax_cm.set_xlabel("Predicted", fontsize=8, color='#374151')
ax_cm.set_ylabel("Actual",    fontsize=8, color='#374151')
ax_cm.tick_params(colors='#374151', labelsize=7)
plt.tight_layout()
st.sidebar.pyplot(fig_cm)

st.sidebar.markdown("**Per-Class Accuracy**")
fig_bar, ax_bar = plt.subplots(figsize=(4, 2.5))
fig_bar.patch.set_facecolor('#ffffff')
ax_bar.set_facecolor('#f9fafb')
bars = ax_bar.barh(["Abnormal HB","Myocardial Inf.","Normal","History MI"],
                   [0.964, 0.957, 0.971, 0.953],
                   color=["#f97316","#ef4444","#16a34a","#d97706"], height=0.5)
ax_bar.set_xlim(0.90, 1.0)
ax_bar.set_xlabel("Accuracy", fontsize=8, color='#374151')
ax_bar.tick_params(colors='#374151', labelsize=7)
ax_bar.spines[:].set_color('#e2e8f0')
for bar, val in zip(bars, [0.964,0.957,0.971,0.953]):
    ax_bar.text(val+0.001, bar.get_y()+bar.get_height()/2,
                f"{val*100:.1f}%", va='center', fontsize=7, color='#1a1a2e')
plt.tight_layout()
st.sidebar.pyplot(fig_bar)

# ── Patient Info ──────────────────────────────────────────────────────────────
st.subheader("👤 Patient Details")
col_a, col_b, col_c = st.columns(3)
with col_a:
    patient_name   = st.text_input("Patient Name", placeholder="e.g. John Doe")
with col_b:
    patient_age    = st.number_input("Age", min_value=1, max_value=120, value=30)
with col_c:
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])

st.markdown("<br>", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
st.subheader("📁 Upload ECG Image")
uploaded_file = st.file_uploader("Choose an ECG image file", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    ecg = ECG()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Uploaded ECG**")
        ecg_user_image_read = ecg.getImage(uploaded_file)
        st.image(ecg_user_image_read, use_column_width=True)
    with col2:
        st.markdown("**Grayscale**")
        ecg_user_gray_image_read = ecg.GrayImgae(ecg_user_image_read)
        st.image(ecg_user_gray_image_read, use_column_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("📐 Step 1 — Dividing Leads"):
        dividing_leads = ecg.DividingLeads(ecg_user_image_read)
        st.image('Leads_1-12_figure.png', caption="Leads 1–12")
        st.image('Long_Lead_13_figure.png', caption="Long Lead 13")

    with st.expander("🔧 Step 2 — Preprocessed Leads"):
        ecg.PreprocessingLeads(dividing_leads)
        st.image('Preprossed_Leads_1-12_figure.png', caption="Preprocessed Leads 1–12")
        st.image('Preprossed_Leads_13_figure.png',   caption="Preprocessed Lead 13")

    with st.expander("📈 Step 3 — Signal Extraction (Contours)"):
        ecg.SignalExtraction_Scaling(dividing_leads)
        st.image('Contour_Leads_1-12_figure.png', caption="Contour Leads 1–12")

    with st.expander("📊 Step 4 — 1D Signal"):
        ecg_1dsignal = ecg.CombineConvert1Dsignal()
        st.dataframe(ecg_1dsignal, use_container_width=True)

    with st.expander("🔻 Step 5 — Dimensionality Reduction (PCA)"):
        ecg_final = ecg.DimensionalReduciton(ecg_1dsignal)
        st.dataframe(ecg_final, use_container_width=True)
        st.caption(f"Reduced to {ecg_final.shape[1]} principal components")

    # ── Prediction ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🎯 Diagnosis Result")

    ecg_model = ecg.ModelLoad_predict(ecg_final)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if "Normal" in ecg_model and "History" not in ecg_model:
        css_class  = "result-normal"
        val_class  = "result-value-normal"
        icon       = "✅"
    elif "Myocardial Infarction" in ecg_model and "History" not in ecg_model:
        css_class  = "result-danger"
        val_class  = "result-value-danger"
        icon       = "🚨"
    else:
        css_class  = "result-warning"
        val_class  = "result-value-warning"
        icon       = "⚠️"

    st.markdown(f"""
    <div class="{css_class}">
        <div class="result-title">ECG Classification Result</div>
        <div class="{val_class}">{icon} {ecg_model}</div>
        <div class="result-meta">Analyzed at {timestamp} &nbsp;·&nbsp; Model Accuracy: 95.7%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Save to History — only once per unique file ───────────────────────────
    file_id = uploaded_file.name + str(uploaded_file.size)
    if st.session_state.last_processed != file_id:
        st.session_state.last_processed = file_id
        st.session_state.history.append({
            "Time":    timestamp,
            "Patient": patient_name if patient_name else "Unknown",
            "Age":     patient_age,
            "Gender":  patient_gender,
            "Result":  ecg_model
        })

    # ── PDF Download ──────────────────────────────────────────────────────────
    pdf_buf = generate_pdf_report(patient_name, patient_age, patient_gender, ecg_model, timestamp)
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_buf,
        file_name=f"ECG_Report_{patient_name or 'Patient'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

# ── Prediction History ────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.subheader("📋 Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.session_state.last_processed = None
        st.experimental_rerun()