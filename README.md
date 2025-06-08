
# Term Deposit Subscription Prediction

This project is part of a **technical assessment** for a Data Analytics role. The objective is to predict whether a client will subscribe to a term deposit based on their profile and previous campaign data.

---

## Project Overview

Using data from direct marketing campaigns by a Portuguese banking institution, we built a machine learning model to predict client subscription outcomes. The primary goal is to help the business optimize outreach strategies and improve campaign success rates.

---

## Dataset

The dataset (`bank-additional-full.csv`) contains 41,188 records and 20+ features, including client demographics, contact campaign details, and economic indicators.

Target variable:  
- `y`: whether the client subscribed to a term deposit (`yes` or `no`)

---

## Steps Taken

1. **Exploratory Data Analysis (EDA)**
   - Visualized class imbalance, outliers, and variable distributions
   - Identified and cleaned placeholder values (`'unknown'`)
   - Detected and handled duplicates and extreme outliers

2. **Feature Engineering**
   - One-hot encoded categorical variables
   - Created new features:
     - `call_efficiency = duration / campaign`
     - `debt_load = 1 if housing or personal loan else 0`
     - `previous_contacted` from `pdays` variable

3. **Model Building**
   - Trained and compared:
     - Logistic Regression with scaling
     - Random Forest with class weight balancing
   - Addressed class imbalance using `class_weight='balanced'`

4. **Model Evaluation**
   - Assessed using accuracy, precision, recall, and F1 score
   - Analyzed confusion matrices and feature importance
   - Checked for overfitting and generalization

---

## Results Summary

| Model              | Train F1 (Yes) | Test F1 (Yes) | Overfitting |
|-------------------|----------------|---------------|-------------|
| Logistic Regression | 0.59           | 0.60          | ❌ No        |
| Random Forest       | 1.00           | 0.53          | ✅ Yes       |

- Logistic Regression offered **balanced performance** and high recall on subscribers.
- Random Forest **overfit** the training data despite strong accuracy.

---

## Key Insights

- **Call duration** and **previous successful contact** are strong predictors.
- Clients with **less debt** and contacted by **cellular phone** are more likely to subscribe.
- **March** and **December** are optimal campaign months.
- **Students, retirees**, and those previously contacted have higher subscription rates.

---

## Recommendation

Deploy the **Logistic Regression model** with engineered features (`call_efficiency`, `debt_load`, `previous_contacted`) to support data-driven marketing. It provides interpretable, stable performance with minimal risk of overfitting.

---

## Files in this Repository

- `term_deposit_subscription_prediction.ipynb`: Main notebook with code, visuals, and commentary
- `README.md`: Project summary
- (Optional) `model_output.csv`: Predictions file (if generated)

---



Ensure your CSV file is located at the specified path in the notebook or adjust accordingly.

---

## Acknowledgements

- Dataset provided by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

---

##Contact

For questions, reach out via email or GitHub Issues.
