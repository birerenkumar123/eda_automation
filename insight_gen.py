import numpy as np


def generate_insights(df):
    insights = []

    # Duplicate check
    if df.duplicated().sum() > 0:
        insights.append(
            f"Dataset contains {df.duplicated().sum()} duplicate rows"
        )

    # Missing values check
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) > 0:
        col = missing.idxmax()
        insights.append(
            f"Highest missing values found in '{col}'"
        )

    # Correlation check
    numeric_cols = df.select_dtypes(include=np.number)

    if not numeric_cols.empty:
        corr = numeric_cols.corr().abs()
        np.fill_diagonal(corr.values, 0)

        max_corr = corr.max().max()

        if max_corr > 0:
            pair = corr.stack().idxmax()
            insights.append(
                f"Strongest correlation between '{pair[0]}' and '{pair[1]}' = {round(max_corr, 2)}"
            )

    return insights