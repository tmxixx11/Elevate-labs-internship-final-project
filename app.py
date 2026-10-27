import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "customer_churn.csv"
DB = ROOT / "data" / "churn.db"

st.set_page_config(page_title="Telecom Customer Churn", page_icon="📉", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(DATA)

def build_model(df):
    X = df.drop(columns=["customer_id", "churn"])
    y = df["churn"].map({"No": 0, "Yes": 1})
    cat = X.select_dtypes(include="object").columns.tolist()
    num = X.select_dtypes(exclude="object").columns.tolist()
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])
    pipe = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]
    scores = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob)
    }
    return pipe, scores, confusion_matrix(y_test, pred, labels=[0, 1])

df = load_data()
with sqlite3.connect(DB) as conn:
    df.to_sql("customers", conn, if_exists="replace", index=False)

model, scores, cm = build_model(df)
st.title("📉 Telecom Customer Churn Analysis")
st.caption("Python • SQL • Scikit-learn • Streamlit")
st.warning("Learning/demo project: the included customer data is synthetic, not real telecom-company data.")

with st.sidebar:
    st.header("Filters")
    contracts = st.multiselect("Contract", sorted(df["contract"].unique()), default=sorted(df["contract"].unique()))
    services = st.multiselect("Internet service", sorted(df["internet_service"].unique()), default=sorted(df["internet_service"].unique()))
    view = df[df.contract.isin(contracts) & df.internet_service.isin(services)]

count = len(view)
churned = int(view.churn.eq("Yes").sum())
a,b,c,d = st.columns(4)
a.metric("Customers", f"{count:,}")
b.metric("Churned", f"{churned:,}")
c.metric("Churn rate", f"{churned/count:.1%}" if count else "—")
d.metric("Average monthly charge", f"${view.monthly_charges.mean():.2f}" if count else "—")

overview, segments, evaluation, predict, sql = st.tabs(["Overview", "Customer segments", "Model evaluation", "Predict churn", "SQL explorer"])
with overview:
    left,right = st.columns(2)
    with left:
        agg = view.groupby("contract", as_index=False).agg(customers=("customer_id","count"), churned=("churn",lambda s:s.eq("Yes").sum()))
        agg["churn_rate"] = agg.churned / agg.customers
        fig = px.bar(agg, x="contract", y="churn_rate", title="Churn rate by contract", text=agg.churn_rate.map(lambda x:f"{x:.1%}"))
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        agg = view.groupby("tenure_months", as_index=False).agg(churn_rate=("churn",lambda s:s.eq("Yes").mean()))
        fig = px.line(agg, x="tenure_months", y="churn_rate", markers=True, title="Churn rate by tenure")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button("Download filtered CSV", view.to_csv(index=False).encode(), "filtered_customers.csv", "text/csv")

with segments:
    st.write("Illustrative segments based on churn labels and tenure; these are rules for this demo, not a validated customer lifecycle model.")
    seg = df.copy()
    seg["segment"] = "Loyal"
    seg.loc[(seg.tenure_months <= 6) & (seg.churn == "No"), "segment"] = "Dormant / New"
    seg.loc[seg.churn == "Yes", "segment"] = "At Risk"
    seg.loc[(seg.tenure_months >= 36) & (seg.churn == "No"), "segment"] = "Loyal"
    summary = seg.groupby("segment", as_index=False).agg(customers=("customer_id","count"))
    fig = px.pie(summary, names="segment", values="customers", title="Customer segment distribution")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(summary, use_container_width=True, hide_index=True)

with evaluation:
    st.write("Logistic regression with a 75/25 train/test split.")
    cols = st.columns(len(scores))
    for col, (name, val) in zip(cols, scores.items()):
        col.metric(name, f"{val:.3f}")
    st.write("Confusion matrix (rows = actual, columns = predicted)")
    st.dataframe(pd.DataFrame(cm, index=["Actual stayed","Actual churned"], columns=["Predicted stayed","Predicted churned"]), use_container_width=True)
    st.caption("Check recall and precision as well as accuracy. Metrics are only a demonstration on synthetic data.")

with predict:
    with st.form("predict"):
        c1,c2,c3 = st.columns(3)
        with c1:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
        with c2:
            internet = st.selectbox("Internet service", ["DSL","Fiber optic","No"])
            payment = st.selectbox("Payment method", ["Electronic check","Mailed check","Bank transfer","Credit card"])
        with c3:
            monthly = st.number_input("Monthly charges", min_value=0.0, max_value=200.0, value=75.0)
            total = st.number_input("Total charges", min_value=0.0, max_value=20000.0, value=900.0)
        submitted = st.form_submit_button("Predict churn")
    if submitted:
        # Build a row with every model feature using representative defaults.
        row = {col: df[col].mode(dropna=True)[0] for col in df.drop(columns=["customer_id","churn"]).columns}
        row.update({"tenure_months":tenure,"contract":contract,"internet_service":internet,"payment_method":payment,"monthly_charges":monthly})
        if "total_charges" in row:
            row["total_charges"] = total
        sample = pd.DataFrame([row], columns=df.drop(columns=["customer_id","churn"]).columns)
        p = float(model.predict_proba(sample)[0,1])
        st.metric("Estimated churn probability", f"{p:.1%}")
        st.write("Higher risk" if p >= .5 else "Lower risk", "under this demonstration model.")
        st.caption("Do not use this synthetic-data model for real customer decisions.")

with sql:
    choices = {
        "Churn rate by contract": """SELECT contract, COUNT(*) AS customers, SUM(CASE WHEN churn='Yes' THEN 1 ELSE 0 END) AS churned, ROUND(100.0*AVG(CASE WHEN churn='Yes' THEN 1.0 ELSE 0.0 END),2) AS churn_rate_pct FROM customers GROUP BY contract ORDER BY churn_rate_pct DESC;""",
        "Average monthly charge by churn": """SELECT churn, COUNT(*) AS customers, ROUND(AVG(monthly_charges),2) AS avg_monthly_charge FROM customers GROUP BY churn;""",
        "Tenure summary": """SELECT MIN(tenure_months) AS min_months, ROUND(AVG(tenure_months),1) AS avg_months, MAX(tenure_months) AS max_months FROM customers;""",
        "Top monthly charges": """SELECT customer_id, contract, monthly_charges, churn FROM customers ORDER BY monthly_charges DESC LIMIT 10;"""
    }
    choice = st.selectbox("Query", list(choices))
    st.code(choices[choice], language="sql")
    with sqlite3.connect(DB) as conn:
        st.dataframe(pd.read_sql_query(choices[choice], conn), use_container_width=True, hide_index=True)
