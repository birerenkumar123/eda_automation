def llm_style_insight(df):
    insights = []

    if df.isnull().sum().sum() > 0:
        insights.append(
            "Dataset contains missing values that may affect model performance"
        )

    if df.duplicated().sum() > 0:
        insights.append(
            "Duplicate rows found and should be removed"
        )

    return insights