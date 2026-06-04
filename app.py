import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBClassifier = None

# ====================== AESTHETIC FULL WINDOW CONFIG ======================
st.set_page_config(
    page_title="Mental Health AI Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Neural Background + Aesthetic Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=Manrope:wght@500;600;700&display=swap');

    :root {
        --heading-font: 'Space Grotesk', sans-serif;
        --body-font: 'Manrope', system-ui, sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0a0f1c 0%, #1a2338 100%);
        color: #e0f2fe;
        font-family: var(--body-font);
    }
    
    .block-container {
        padding-top: 1.5rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        max-width: 1420px !important;
        margin: 0 auto;
    }
    
    #neural-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        opacity: 0.2;
        pointer-events: none;
    }
    
    .compact-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(90deg, #1e2937, #334155);
        padding: 1.4rem 3rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        box-shadow: 0 15px 40px rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(165, 243, 252, 0.25);
    }
    
    .header-title {
        font-family: var(--heading-font);
        background: linear-gradient(90deg, #67e8f9, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.7rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -2px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15, 23, 42, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 8px;
        margin-bottom: 2rem;
        border: 1px solid rgba(165, 243, 252, 0.25);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 41, 59, 0.85);
        border-radius: 16px;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 1.05rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #22d3ee, #c084fc) !important;
        color: white !important;
    }
    
    .card {
        background: rgba(15, 23, 42, 0.85);
        border-radius: 22px;
        padding: 24px;
        border: 1px solid rgba(165, 243, 252, 0.2);
        box-shadow: 0 15px 38px rgba(0, 0, 0, 0.4);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e2937, #0f172a);
        border-radius: 20px;
        padding: 22px 16px;
        text-align: center;
        border: 1px solid #334155;
        height: 130px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .model-explain {
        background: linear-gradient(135deg, #1e2937, #334155);
        padding: 18px 24px;
        border-radius: 20px;
        border-left: 6px solid #67e8f9;
        font-size: 0.97rem;
        margin-bottom: 1.4rem;
    }
    
    .section-title {
        color: #bae6fd;
        font-size: 1.65rem;
        margin: 1rem 0 1.5rem 0;
        border-bottom: 3px solid #67e8f9;
        padding-bottom: 12px;
        font-family: var(--heading-font);
    }
</style>

<canvas id="neural-bg"></canvas>

<script>
    const canvas = document.getElementById('neural-bg');
    const ctx = canvas.getContext('2d');
    let width, height, particles = [];
    
    class Particle {
        constructor() {
            this.x = Math.random() * width;
            this.y = Math.random() * height;
            this.size = Math.random() * 2.2 + 0.8;
            this.speedX = Math.random() * 0.6 - 0.3;
            this.speedY = Math.random() * 0.6 - 0.3;
        }
        update() {
            this.x += this.speedX;
            this.y += this.speedY;
            if (this.x < 0 || this.x > width) this.speedX *= -1;
            if (this.y < 0 || this.y > height) this.speedY *= -1;
        }
        draw() {
            ctx.fillStyle = 'rgba(103, 232, 249, 0.9)';
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
            ctx.fill();
        }
    }
    
    function resize() {
        width = window.innerWidth;
        height = window.innerHeight;
        canvas.width = width;
        canvas.height = height;
    }
    
    function connect() {
        for (let a = 0; a < particles.length; a++) {
            for (let b = a; b < particles.length; b++) {
                const dx = particles[a].x - particles[b].x;
                const dy = particles[a].y - particles[b].y;
                const distance = Math.hypot(dx, dy);
                if (distance < 150) {
                    ctx.strokeStyle = `rgba(103, 232, 249, ${0.18 * (150 - distance) / 150})`;
                    ctx.lineWidth = 1.1;
                    ctx.beginPath();
                    ctx.moveTo(particles[a].x, particles[a].y);
                    ctx.lineTo(particles[b].x, particles[b].y);
                    ctx.stroke();
                }
            }
        }
    }
    
    function animate() {
        ctx.clearRect(0, 0, width, height);
        for (let i = 0; i < particles.length; i++) {
            particles[i].update();
            particles[i].draw();
        }
        connect();
        requestAnimationFrame(animate);
    }
    
    function init() {
        resize();
        particles = [];
        const count = Math.floor(Math.min(width * height / 11000, 100));
        for (let i = 0; i < count; i++) particles.push(new Particle());
        animate();
    }
    
    window.addEventListener('resize', init);
    window.onload = init;
</script>
""", unsafe_allow_html=True)

# ====================== HEADER ======================
col_header1, col_header2 = st.columns([3, 1])
with col_header1:
    st.markdown('<h1 class="header-title">🧠 Mental Health AI Dashboard</h1>', unsafe_allow_html=True)
with col_header2:
    st.markdown('<p style="text-align:right; color:#64748b; margin-top:32px; font-size:1.05rem; font-weight:500;">Professional ML Intelligence Platform</p>', unsafe_allow_html=True)

st.markdown('<p style="color:#94a3b8; margin-top:-8px; margin-bottom:2rem; font-size:1.1rem; font-weight:500;">Supervised • Unsupervised • Real-time Risk Intelligence</p>', unsafe_allow_html=True)

# ====================== TABS ======================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Overview", 
    "🔎 Explorer", 
    "📊 Visuals", 
    "🧪 ML Arena", 
    "🔬 Clustering", 
    "🔮 Predictor"
])

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("data/Student_Mental_Health.csv")
    df.rename(columns={
        "Choose your gender": "gender", "Age": "age", "What is your course?": "course",
        "Your current year of Study": "year", "What is your CGPA?": "cgpa",
        "Marital status": "marital_status", "Do you have Depression?": "depression",
        "Do you have Anxiety?": "anxiety", "Do you have Panic attack?": "panic_attack",
        "Did you seek any specialist for a treatment?": "treatment"
    }, inplace=True)
    df = df.dropna().copy()
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    for col in ['depression', 'anxiety', 'panic_attack']:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}).fillna(0)
    df['mental_score'] = df[['depression', 'anxiety', 'panic_attack']].sum(axis=1)
    return df

df = load_data()

with st.sidebar:
    st.subheader("Live Filters")
    gender_filter = st.multiselect("Gender", df['gender'].unique(), default=df['gender'].unique())
    year_filter = st.multiselect("Year", sorted(df['year'].unique()), default=df['year'].unique())
    filtered_df = df[(df['gender'].isin(gender_filter)) & (df['year'].isin(year_filter))]

# ====================== TAB 1: OVERVIEW ======================
with tab1:
    st.markdown('<p class="section-title">Dashboard Overview</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4, gap="small")
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div style="color:#94a3b8; font-size:0.9rem; font-weight:500;">TOTAL STUDENTS</div>
                <div style="font-size:2.8rem; font-weight:700; color:#bae6fd; margin-top:6px;">{len(filtered_df)}</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div style="color:#94a3b8; font-size:0.9rem; font-weight:500;">DEPRESSION RATE</div>
                <div style="font-size:2.8rem; font-weight:700; color:#f87171; margin-top:6px;">{filtered_df['depression'].mean():.1%}</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div style="color:#94a3b8; font-size:0.9rem; font-weight:500;">ANXIETY RATE</div>
                <div style="font-size:2.8rem; font-weight:700; color:#fb923c; margin-top:6px;">{filtered_df['anxiety'].mean():.1%}</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div style="color:#94a3b8; font-size:0.9rem; font-weight:500;">AVG MENTAL SCORE</div>
                <div style="font-size:2.8rem; font-weight:700; color:#67e8f9; margin-top:6px;">{filtered_df['mental_score'].mean():.2f}<span style="font-size:1.2rem; color:#64748b;">/3</span></div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig = px.histogram(filtered_df, x="course", color="mental_score", title="Mental Score Distribution by Course", height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig = px.pie(filtered_df, names="mental_score", title="Overall Mental Health Distribution", height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<p class="section-title">Data Explorer</p>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.dataframe(filtered_df, use_container_width=True, height=520)
    st.markdown('</div>', unsafe_allow_html=True)
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Current Data", csv, "mental_health_data.csv", "text/csv", use_container_width=True)

with tab3:
    st.markdown('<p class="section-title">Advanced Visualizations</p>', unsafe_allow_html=True)
    sub1, sub2 = st.tabs(["Distribution Analysis", "Correlation Matrix"])
    with sub1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            fig = px.box(filtered_df, x="gender", y="mental_score", color="gender", title="Mental Score by Gender")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            fig = px.violin(filtered_df, x="year", y="mental_score", title="Mental Score by Academic Year")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    with sub2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        numeric_cols = filtered_df.select_dtypes(include=np.number).columns
        corr = filtered_df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, linewidths=0.6)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<p class="section-title">Supervised Machine Learning Arena</p>', unsafe_allow_html=True)
    st.markdown('<div class="model-explain">Supervised models untuk memprediksi depresi. Termasuk Random Forest, XGBoost (jika tersedia), Logistic Regression, Decision Tree, dan KNN dengan evaluasi lengkap.</div>', unsafe_allow_html=True)
    
    X = filtered_df[['age', 'mental_score']]
    y = filtered_df['depression']
    
    if len(y.unique()) < 2:
        st.error("⚠️ Data setelah filter hanya memiliki 1 kelas depresi. Model tidak bisa dilatih. Silakan ubah filter di sidebar agar ada variasi data (baik yang depresi maupun tidak).")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
            "KNN": KNeighborsClassifier()
        }
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBClassifier(eval_metric='logloss', random_state=42, n_estimators=100)
        
        results = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)
            prec = precision_score(y_test, pred, zero_division=0)
            rec = recall_score(y_test, pred, zero_division=0)
            f1 = f1_score(y_test, pred, zero_division=0)
            results.append([name, round(acc*100, 2), round(prec*100, 2), round(rec*100, 2), round(f1*100, 2)])
        
        result_df = pd.DataFrame(results, columns=["Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"])
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        
        fig = px.bar(result_df, x="Model", y="Accuracy (%)", title="Model Performance Comparison", color="Accuracy (%)", height=380)
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.markdown('<p class="section-title">Unsupervised Machine Learning</p>', unsafe_allow_html=True)
    st.markdown('<div class="model-explain">K-Means Clustering + PCA untuk menemukan pola tersembunyi pada kesehatan mental mahasiswa.</div>', unsafe_allow_html=True)
    
    features = filtered_df[['age', 'mental_score', 'depression', 'anxiety', 'panic_attack']]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features)
    temp_df = filtered_df.copy()
    temp_df['Cluster'] = clusters.astype(str)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig = px.scatter(temp_df, x="age", y="mental_score", color="Cluster", title="K-Means Student Clusters")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features)
        temp_df['PCA1'] = pca_result[:, 0]
        temp_df['PCA2'] = pca_result[:, 1]
        fig2 = px.scatter(temp_df, x="PCA1", y="PCA2", color="Cluster", title="PCA 2D Projection")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab6:
    st.markdown('<p class="section-title">Real-time Risk Prediction</p>', unsafe_allow_html=True)
    st.markdown('<div class="model-explain">Random Forest Classifier digunakan untuk memprediksi risiko depresi secara real-time.</div>', unsafe_allow_html=True)
    
    X = filtered_df[['age', 'mental_score']]
    y = filtered_df['depression']
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    
    col_p1, col_p2 = st.columns([1, 2])
    with col_p1:
        age = st.slider("Umur Mahasiswa", 17, 35, 21)
        mental_score = st.slider("Mental Health Score", 0, 3, 1)
        if st.button("🚀 Predict Risk Level", type="primary", use_container_width=True):
            pred = model.predict([[age, mental_score]])[0]
            risk = (mental_score / 3) * 100
            st.metric("Risk Level", f"{risk:.1f}%")
            if pred == 1 or risk > 65:
                st.error("🔴 HIGH RISK — Immediate professional help recommended")
            elif risk > 35:
                st.warning("🟠 MODERATE RISK — Consider counseling")
            else:
                st.success("🟢 LOW RISK — Maintain positive habits")

st.caption("Mental Health AI Dashboard • Fixed Class Error + Aesthetic Design • Ogy dev")
