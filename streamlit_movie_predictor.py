# ============================================================
# Movie Revenue Predictor — Production Streamlit Application
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Movie Revenue Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# GLOBAL STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        font-weight: 700;
    }
    .metric-box {
        background: #161b22;
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #30363d;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# MODEL ANALYZER CLASS
# ============================================================

class TMDBPredictiveAnalyzer:
    def __init__(self, model_path: str):
        loaded = joblib.load(model_path)

        self.model = loaded["model"]
        self.preprocessor = loaded["preprocessor"]
        self.feature_names = loaded["feature_names"]
        self.metadata = loaded.get("metadata", {})

        self._patch_imputers()

    def _patch_imputers(self):
        """
        Backward compatibility patch for sklearn >=1.7
        Fixes SimpleImputer deserialization errors.
        """
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer

        def patch(obj):
            if isinstance(obj, SimpleImputer):
                if not hasattr(obj, "_fill_dtype"):
                    obj._fill_dtype = None

        if isinstance(self.model, Pipeline):
            for _, step in self.model.steps:
                patch(step)

        if isinstance(self.preprocessor, ColumnTransformer):
            for _, transformer, _ in self.preprocessor.transformers_:
                if isinstance(transformer, Pipeline):
                    for _, step in transformer.steps:
                        patch(step)
                else:
                    patch(transformer)

    def predict_revenue(self, **features) -> float:
        X = pd.DataFrame([features], columns=self.feature_names)
        log_pred = self.model.predict(X)[0]
        return float(np.expm1(log_pred))

    def analyze_roi(self, budget: float, **features) -> dict:
        revenue = self.predict_revenue(**features)
        profit = revenue - budget
        roi = (profit / budget) * 100 if budget > 0 else 0

        return {
            "revenue": revenue,
            "profit": profit,
            "roi": roi
        }

# ============================================================
# LOAD MODEL (SAFE)
# ============================================================

MODEL_PATH = Path("model/movie_revenue_model.pkl")

@st.cache_resource(show_spinner=True)
def load_analyzer():
    return TMDBPredictiveAnalyzer(MODEL_PATH)

analyzer = load_analyzer()

# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.title("🎬 Movie Parameters")

budget = st.sidebar.number_input(
    "Production Budget (USD)",
    min_value=1_000_000,
    max_value=500_000_000,
    value=50_000_000,
    step=1_000_000
)

runtime = st.sidebar.slider("Runtime (minutes)", 60, 240, 120)
popularity = st.sidebar.slider("Popularity Index", 0.0, 100.0, 40.0)
vote_average = st.sidebar.slider("Average Rating", 0.0, 10.0, 6.5)
vote_count = st.sidebar.number_input("Vote Count", 100, 500_000, 15_000)

release_month = st.sidebar.selectbox(
    "Release Month",
    list(range(1, 13)),
    index=5
)

is_holiday_release = st.sidebar.radio(
    "Holiday Release",
    ["No", "Yes"],
    horizontal=True
)

# ============================================================
# FEATURE DICTIONARY (MUST MATCH TRAINING)
# ============================================================

features = {
    "budget": budget,
    "runtime": runtime,
    "popularity": popularity,
    "vote_average": vote_average,
    "vote_count": vote_count,
    "release_month": release_month,
    "holiday_release": 1 if is_holiday_release == "Yes" else 0
}

# ============================================================
# MAIN DASHBOARD
# ============================================================

st.title("🎥 Movie Revenue Prediction Dashboard")
st.caption("AI-driven box office forecasting and ROI analysis")

st.markdown("---")

# ============================================================
# RUN ANALYSIS
# ============================================================

analysis = analyzer.analyze_roi(budget=budget, **features)

# ============================================================
# KPI METRICS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="metric-box">
            <h3>Predicted Revenue</h3>
            <h2>${analysis['revenue']:,.0f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-box">
            <h3>Expected Profit</h3>
            <h2>${analysis['profit']:,.0f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-box">
            <h3>ROI</h3>
            <h2>{analysis['roi']:.1f}%</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# VISUAL ANALYSIS
# ============================================================

st.markdown("### 📊 Budget vs Predicted Revenue")

fig = go.Figure()

fig.add_bar(
    x=["Budget", "Predicted Revenue"],
    y=[budget, analysis["revenue"]],
)

fig.update_layout(
    height=420,
    template="plotly_dark",
    yaxis_title="USD",
    xaxis_title="",
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# INTERPRETATION
# ============================================================

st.markdown("### 🧠 Model Interpretation")

if analysis["roi"] > 50:
    verdict = "High-potential investment with strong upside."
elif analysis["roi"] > 0:
    verdict = "Moderate return expected; marketing execution is critical."
else:
    verdict = "High financial risk; budget optimization advised."

st.info(verdict)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "Developed by Trymore Mhlanga | Machine Learning • Predictive Analytics • Streamlit"
)
