# Loan_defaulter - Credit Risk Default Prediction


End-to-end ETL Pipeline, Machine learning model using random forest model that that estimates the likelihood a borrower will default on a loan. Built with Python, SQL (SQLite) and scikit-learn.



## Overview
- **Extract:** reads 32,581 loan records from a raw CSV, unchanged
- **Transform:** removes duplicates and impossible values (e.g. age > 100), and quarantines the rejected rows
- **Load:** writes clean and rejected data to a SQLite database (idempotent, safe to re-run)
- **Explore (SQL):** compares default rates by loan grade, home ownership and loan purpose
- **Model:** a random forest classifier in a scikit-learn pipeline (imputation and encoding happen inside the pipeline to avoid data leakage)
- **Tune:** the decision threshold is chosen from the precision-recall trade-off, instead of the default 0.5

## Results
At a 0.35 threshold on a held-out test set:
- Catches **84% of defaults** (recall) with **63% precision**
- Default rate among approved loans drops from about **22% to about 5%**
- The cost is declining about 14% of good applicants, a trade-off driven by the cost of a missed default

## Tech
Python, pandas, SQL (SQLite), scikit-learn, Google Colab

## Key takeaways
- Loan grade, home ownership and loan purpose are strongly linked to default
- Accuracy is misleading with imbalanced classes (about 22% defaults), so the model is judged on AUC, precision and recall
- Limitations: the data contains only approved loans (selection bias), and the threshold was tuned on the test set
