# ================================
# STREAMLIT MOVIE REVENUE ANALYTICS
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Trymore's Movie Revenue Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- MODEL CLASS ----------------
class TMDBPredictiveAnalyzer:
    def __init__(self, model_path):
        loaded = joblib.load(model_path)
        self.model = loaded["model"]
        self.preprocessor = loaded["preprocessor"]
        self.feature_names = loaded["feature_names"]
        self.model_metadata = loaded.get("metadata", {})

    def _prepare_input(
        self,
        budget,
        runtime,
        vote_average,
        vote_count,
        release_month,
        release_year,
        primary_genre
    ):
        base = pd.DataFrame([{
            "budget": float(budget),
            "runtime": int(runtime),
            "vote_average": float(vote_average),
            "vote_count": int(vote_count),
            "release_month": int(release_month),
            "release_year": int(release_year)
        }])

        base["budget_log"] = np.log1p(base["budget"])
        base["vote_score"] = base["vote_average"] * base["vote_count"]
        base["is_summer_release"] = base["release_month"].isin([5,6,7,8]).astype(int)
        base["is_holiday_release"] = base["release_month"].isin([11,12]).astype(int)
        base["release_quarter"] = ((base["release_month"] - 1) // 3 + 1).astype(int)

        X = pd.DataFrame(0.0, columns=self.feature_names, index=[0])

        for col in base.columns:
            if col in X.columns:
                X[col] = base[col].astype(float)

        genre_col = f"genre_{primary_genre}"
        if genre_col in X.columns:
            X[genre_col] = 1.0

        return X

    def predict_revenue(self, **params):
        X = self._prepare_input(**params)
        log_pred = self.model.predict(X)[0]
        return float(np.expm1(log_pred))

    def analyze_roi(self, **params):
        revenue = self.predict_revenue(**params)
        budget = float(params["budget"])

        profit = revenue - budget
        roi = (profit / budget) * 100 if budget > 0 else 0

        return {
            "predicted_revenue": revenue,
            "investment": budget,
            "profit": profit,
            "roi_percentage": roi,
            "profit_margin": (profit / revenue) * 100 if revenue > 0 else 0,
            "risk_level": "Low" if roi > 50 else "Medium" if roi > 0 else "High",
            "risk_factors": [],
            "breakeven_multiplier": revenue / budget if budget > 0 else 0,
            "confidence_score": min(100, params["vote_average"] * 10)
        }

    # -------- FIXED & REQUIRED METHOD --------
    def optimize_release_timing(
        self,
        budget,
        runtime,
        vote_average,
        vote_count,
        release_year,
        primary_genre
    ):
        monthly = {}

        for m in range(1, 13):
            params = dict(
                budget=budget,
                runtime=runtime,
                vote_average=vote_average,
                vote_count=vote_count,
                release_year=release_year,
                release_month=m,
                primary_genre=primary_genre
            )

            result = self.analyze_roi(**params)
            monthly[m] = result

        best_month = max(monthly, key=lambda m: monthly[m]["roi_percentage"])

        return {
            "best_month": best_month,
            "best_roi": monthly[best_month]["roi_percentage"],
            "monthly_analysis": monthly,
            "seasonal_insights": {
                "best_season": "Summer" if best_month in [6,7,8] else "Other",
                "seasonal_averages": {
                    "Q1": np.mean([monthly[m]["roi_percentage"] for m in [1,2,3]]),
                    "Q2": np.mean([monthly[m]["roi_percentage"] for m in [4,5,6]]),
                    "Q3": np.mean([monthly[m]["roi_percentage"] for m in [7,8,9]]),
                    "Q4": np.mean([monthly[m]["roi_percentage"] for m in [10,11,12]])
                }
            }
        }

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_analyzer():
    return TMDBPredictiveAnalyzer("models/tmdb_analyzer.pkl")

analyzer = load_analyzer()

# ---------------- SIDEBAR ----------------
st.sidebar.header("Movie Configuration")

params = {
    "budget": st.sidebar.slider("Budget", 100_000, 300_000_000, 50_000_000, step=1_000_000),
    "runtime": st.sidebar.slider("Runtime (minutes)", 60, 240, 120),
    "vote_average": st.sidebar.slider("TMDB Rating", 1.0, 10.0, 7.0, 0.1),
    "vote_count": st.sidebar.slider("Vote Count", 100, 10_000, 2_000, 100),
    "primary_genre": st.sidebar.selectbox(
        "Genre",
        ["Action","Adventure","Animation","Comedy","Crime","Drama",
         "Fantasy","Horror","Romance","Thriller"]
    ),
    "release_month": st.sidebar.selectbox("Release Month", list(range(1,13)), index=6),
    "release_year": st.sidebar.number_input("Release Year", 2024, 2030, 2024)
}

# ---------------- MAIN ----------------
st.title("Movie Revenue Analytics Platform")

if st.sidebar.button("Run Analysis"):
    analysis = analyzer.analyze_roi(**params)
    timing = analyzer.optimize_release_timing(
        budget=params["budget"],
        runtime=params["runtime"],
        vote_average=params["vote_average"],
        vote_count=params["vote_count"],
        release_year=params["release_year"],
        primary_genre=params["primary_genre"]
    )

    st.metric("Predicted Revenue", f"${analysis['predicted_revenue']:,.0f}")
    st.metric("ROI", f"{analysis['roi_percentage']:.1f}%")
    st.metric("Best Release Month", timing["best_month"])
