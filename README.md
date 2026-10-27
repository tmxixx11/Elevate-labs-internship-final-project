# Telecom Customer Churn Analysis

## Open in VS Code
1. Extract this ZIP and open the `customer_churn_vscode` folder in VS Code.
2. Open Terminal → New Terminal.
3. Install dependencies (Python 3.11 recommended):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

4. Open the local URL printed by Streamlit (usually http://localhost:8501).

## Included
- `app.py`: interactive dashboard, churn model, segments, prediction form and SQL explorer
- `analysis.py`: quick analysis in the terminal
- `sql_queries.sql`: example SQL queries
- `data/customer_churn.csv`: sample dataset
- `requirements.txt`: required packages

The included dataset is synthetic for learning; it is not real telecom customer data. Model results are illustrative and should not be used to make decisions about real customers.
