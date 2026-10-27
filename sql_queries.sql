-- Run after app.py has created data/churn.db and the customers table.

SELECT ROUND(100.0 * AVG(CASE WHEN churn='Yes' THEN 1.0 ELSE 0.0 END), 2) AS churn_rate_pct
FROM customers;

SELECT contract, COUNT(*) AS customers,
       SUM(CASE WHEN churn='Yes' THEN 1 ELSE 0 END) AS churned,
       ROUND(100.0 * AVG(CASE WHEN churn='Yes' THEN 1.0 ELSE 0.0 END), 2) AS churn_rate_pct
FROM customers GROUP BY contract ORDER BY churn_rate_pct DESC;

SELECT churn, ROUND(AVG(monthly_charges), 2) AS avg_monthly_charge
FROM customers GROUP BY churn;

SELECT payment_method, COUNT(*) AS customers,
       ROUND(100.0 * AVG(CASE WHEN churn='Yes' THEN 1.0 ELSE 0.0 END), 2) AS churn_rate_pct
FROM customers GROUP BY payment_method ORDER BY churn_rate_pct DESC;
