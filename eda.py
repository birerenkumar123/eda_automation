import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier


def basic_info(df, st):
    st.subheader("Basic Information")
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])
    st.write("Duplicate Rows:", df.duplicated().sum())

    dtype_df = df.dtypes.reset_index()
    dtype_df.columns = ["Column", "Datatype"]
    st.dataframe(dtype_df)


def missing_values(df, st):
    st.subheader("Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) > 0:
        missing_df = missing.reset_index()
        missing_df.columns = ["Column", "Missing Count"]
        st.dataframe(missing_df)
    else:
        st.success("No Missing Values")


def statistical_summary(df, st):
    st.subheader("Statistical Analysis")
    numeric_cols = df.select_dtypes(include=np.number)

    if not numeric_cols.empty:
        st.write("Describe Statistics")
        st.dataframe(numeric_cols.describe())

        st.write("Skewness")
        st.dataframe(numeric_cols.skew().reset_index().rename(columns={"index": "Column", 0: "Skewness"}))

        st.write("Kurtosis")
        st.dataframe(numeric_cols.kurt().reset_index().rename(columns={"index": "Column", 0: "Kurtosis"}))


def correlation_analysis(df, st):
    numeric_cols = df.select_dtypes(include=np.number)

    if not numeric_cols.empty:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)


def outlier_detection(df, st):
    numeric_cols = df.select_dtypes(include=np.number)

    if not numeric_cols.empty:
        st.subheader("Outlier Detection")

        for col in numeric_cols.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outliers = df[(df[col] < lower) | (df[col] > upper)]
            st.write(f"{col}: {len(outliers)} outliers")


def visualization_tools(df, st):
    st.subheader("Visualization Tools")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    if numeric_cols:
        selected_num = st.selectbox("Select Numeric Column", numeric_cols)

        st.write("Histogram + Distplot")
        fig, ax = plt.subplots()
        sns.histplot(df[selected_num].dropna(), kde=True, ax=ax)
        st.pyplot(fig)

        st.write("Box Plot")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[selected_num], ax=ax)
        st.pyplot(fig)

        st.write("Line Plot")
        fig, ax = plt.subplots()
        plt.plot(df[selected_num])
        st.pyplot(fig)

        st.write("QQ Plot")
        fig, ax = plt.subplots()
        stats.probplot(df[selected_num].dropna(), dist="norm", plot=ax)
        st.pyplot(fig)

    if categorical_cols:
        selected_cat = st.selectbox("Select Categorical Column", categorical_cols)

        st.write("Bar Plot")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(x=df[selected_cat], order=df[selected_cat].value_counts().index, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)


def data_quality_score(df):
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()

    quality_score = 100 - (
        (missing_cells / total_cells) * 50 +
        (duplicate_rows / df.shape[0]) * 50
    )

    return round(quality_score, 2)


def class_imbalance_check(df, target, st):
    st.subheader("Class Imbalance Detection")

    class_counts = df[target].value_counts()

    st.dataframe(class_counts)

    st.bar_chart(class_counts)


def feature_recommendation(df):
    recommendations = []

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            recommendations.append(f"Handle missing values in {col}")

        if df[col].dtype == "object":
            recommendations.append(f"Apply encoding for categorical column {col}")

    return recommendations
def feature_recommendation(df):
    recommendations = []

    for col in df.columns:

        # Missing values recommendation
        missing_count = df[col].isnull().sum()
        missing_percent = (missing_count / len(df)) * 100

        if missing_count > 0:
            recommendations.append(
                f"{col} → Missing Values: {missing_count} ({round(missing_percent, 2)}%) → Handle missing values"
            )

        # Categorical column recommendation
        if df[col].dtype == "object":
            unique_values = df[col].nunique()

            if unique_values < 15:
                encode_type = "Label Encoding"

            else:
                encode_type = "One-Hot Encoding"

            recommendations.append(
                f"{col} → Categorical Column ({unique_values} unique values) → Apply {encode_type}"
            )

        # High cardinality warning
        if df[col].nunique() > 100:
            recommendations.append(
                f"{col} → High Cardinality Column → Consider Feature Reduction"
            )

        # Constant column check
        if df[col].nunique() == 1:
            recommendations.append(
                f"{col} → Constant Column → Consider Dropping"
            )

    return recommendations


def suggest_ml_model(target_series):
    if target_series.dtype == "object":
        return "Classification Problem → Use Logistic Regression, Random Forest"

    elif target_series.nunique() < 10:
        return "Classification Problem → Use Tree Models"

    else:
        return "Regression Problem → Use Linear Regression, XGBoost"
def feature_importance(df, target, st):
    df = df.dropna()

    X = df.drop(columns=[target])
    y = df[target]

    X = pd.get_dummies(X, drop_first=True)

    model = RandomForestClassifier()
    model.fit(X, y)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    st.subheader("Feature Importance")
    st.dataframe(importance.head(15))