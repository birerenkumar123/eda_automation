import streamlit as st
import pandas as pd

from insight_gen import generate_insights
from llm_insight import llm_style_insight
from report_generator import create_pdf_report

from eda import (
    basic_info,
    missing_values,
    statistical_summary,
    correlation_analysis,
    outlier_detection,
    visualization_tools,
    data_quality_score,
    feature_recommendation,
    suggest_ml_model,
)

st.set_page_config(
    page_title="EDA Automation",
    layout="wide"
)

st.markdown("""
# Intelligent Automated EDA Insight Generator
### Upload CSV dataset and get automatic EDA + ML Insights
""")

st.sidebar.title("EDA Dashboard")
st.sidebar.write(
    "Professional EDA Project for Data Analyst / Data Science Roles"
)

# -------------------------
# CSV Upload Only
# -------------------------

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

df = None
insights = []

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

# -------------------------
# Main Processing
# -------------------------

if df is not None:

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        score = data_quality_score(df)
        st.metric("Data Quality Score", f"{score}%")

    # EDA Functions
    basic_info(df, st)
    missing_values(df, st)
    statistical_summary(df, st)
    correlation_analysis(df, st)
    outlier_detection(df, st)
    visualization_tools(df, st)

    # Business Insights
    st.subheader("Business Insights")
    insights = generate_insights(df)

    for item in insights:
        st.success(item)

    # LLM-Based Insights
    st.subheader("LLM-Based Smart Insights")
    llm_insights = llm_style_insight(df)

    for item in llm_insights:
        st.warning(item)

    # Feature Engineering Recommendations
    st.subheader("Feature Engineering Recommendations")
    recommendations = feature_recommendation(df)

    if recommendations:
        for rec in recommendations:
            st.info(rec)
    else:
        st.success(
            "No major preprocessing recommendations needed"
        )

    # ML Model Suggestion
    st.subheader("ML Model Suggestion")

    target_column = st.selectbox(
        "Select Target Column",
        df.columns
    )

    if target_column:
        model_suggestion = suggest_ml_model(
            df[target_column]
        )
        st.warning(model_suggestion)

    # PDF Report Export
    st.subheader("PDF Report Export")

    if st.button(
        "Generate PDF Report",
        key="pdf_button"
    ):
        create_pdf_report(insights)
        st.success(
            "PDF Report Generated Successfully"
        )

else:
    st.info(
        "Please upload CSV file to start analysis"
    )