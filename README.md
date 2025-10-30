# BigData-Inclass-Assignment
# Customer Churn Prediction with PySpark MLlib

This project demonstrates a comprehensive machine learning workflow using **Apache Spark's MLlib** to predict customer churn. The analysis begins with exploratory data analysis (EDA), followed by data preprocessing, feature engineering, model training, and hyperparameter tuning to identify the best-performing classification model.

---

## 🎯 Project Objective

Build, evaluate, and compare multiple machine learning models using PySpark to predict whether a customer will churn based on their service usage, subscription details, and satisfaction scores. The project emphasizes not just model building but also interpretation and deriving business insights.

---

## 💾 Dataset

The dataset used is `churn_streaming_data.csv`, which contains various customer attributes such as age, region, plan type, monthly fee, tenure, and satisfaction scores. The target variable is `churn_flag`, where **1 indicates churn** and **0 indicates no churn**.
**Query To generate the dataset Synthetically:**
```
Act as a data scientist for "StreamNow," a subscription streaming service. We are trying to build a predictive model to identify customers who are at high risk of canceling their subscription in the next month. This is critical for our business because we want to proactively target these at-risk customers with special retention offers, like discounts or content promotions, to reduce overall churn and protect our revenue.

Please generate a synthetic dataset for this task with the following specifications:

Prediction Objective:
The target variable should be named churn_flag. It must be a binary classification variable where 1 means the customer churned (canceled) and 0 means the customer did not churn.

Feature List (12 features + 1 ID):
Generate data with the following features, types, and relationships:
1. customer_id: A unique string identifier for each customer (e.g., "CUST-1001").
2. age: Customer's age (numeric, between 18 and 75).
3. region: Geographic region (categorical: "North", "South", "East", "West").
4. plan_type: Subscription plan (categorical: "Basic", "Standard", "Premium").
5. monthly_fee: The numeric dollar amount of the monthly subscription ($8.99, $15.49, $19.99). Make this fee correspond logically to the plan_type.
6. tenure_months: How many months the customer has been subscribed (numeric, 1 to 60).
7. logins_per_week: Average number of logins per week (numeric, 0 to 20).
8. avg_session_minutes: Average duration of a viewing session in minutes (numeric, 5 to 240).
9. content_watched_per_week: Average number of movies/shows watched per week (numeric, 0 to 30).
10. num_support_tickets: Number of support tickets filed in the last 6 months (numeric, 0 to 10).
11. satisfaction_score: A customer satisfaction score from a recent survey (ordinal, 1 to 5, where 1 is very dissatisfied and 5 is very satisfied).
12. used_discount: A flag indicating if the customer is currently using a promotional discount (binary: 1 for yes, 0 for no).

Correlations & Relationships:
Please ensure the data reflects realistic business scenarios by including these correlations with the churn_flag:
* Higher churn_flag (more likely to churn) should be correlated with:
    1. Higher num_support_tickets.
    2. Lower tenure_months.
    3. Lower logins_per_week.
    4. Lower satisfaction_score.
    5. Lower avg_session_minutes.
* Lower churn_flag (less likely to churn) should be correlated with:
    1. Higher tenure_months.
    2. Higher satisfaction_score.
    3. Being on a "Premium" plan_type.
    4. Higher content_watched_per_week.

Data Size and Format:
Generate 1,000 rows of this synthetic data and provide it in *CSV format*. The first row must contain the headers as specified above.

Data Quality Simulation: To simulate messy, real-world data, please include a small amount of missing values (less than 5%) in the satisfaction_score and avg_session_minutes columns only.
```
---

## 📂 Output Files

The script will generate the following directory structure and files:
```bash
outputs/
├── task0/
│   └── task0_data_exploration.txt
├── task1/
│   └── task1_preprocessing_summary.txt
├── task2/
│   └── task2_logistic_regression_results.txt
│   └── ... (plots)
├── task3/
│   └── task3_feature_selection.txt
├── task4/
│   └── task4_model_comparison.txt
└── task5/
    └── task5_business_takeaway.txt
```

---

## 🚀 Project Workflow

### Task 0: Exploratory Data Analysis (EDA)
**Objective:** Understand the dataset's structure, distributions, and relationships before modeling.

**Explanation of Output:**
The EDA summary provides a foundational understanding of the dataset. It shows that we are working with **216 customer records** and **12 features**. The descriptive statistics reveal key insights, such as an average customer age of 45 and an average tenure of about 27 months. The mean churn rate is approximately 33%, indicating a moderate class imbalance.

**Output (`outputs/task0/task0_data_exploration.txt`):**
```bash
Task 0: Data Import and Exploratory Data Analysis

--- 1. Data Import ---
Number of rows: 216
Number of columns: 12

--- 2. Summary Statistics ---
                      age  monthly_fee  tenure_months  ...  satisfaction_score  used_discount  churn_flag
count                216   216.000000     216.000000  ...          213.000000     215.000000  215.000000
mean   45.569444   14.395093      27.606481  ...            3.276995       0.372093    0.330233
std    15.044578    4.522196      19.177452  ...            1.396343       0.484559    0.471376
min    18.000000    8.990000       1.000000  ...            1.000000       0.000000    0.000000
max    70.000000   19.990000      60.000000  ...            5.000000       1.000000    1.000000
```

---

### Task 1: Data Preprocessing and Feature Engineering
**Objective:** Transform raw data into a format suitable for machine learning algorithms.

**Explanation of Output:**
This output shows the result of the preprocessing pipeline. Each row now consists of a single dense vector `(features)` that combines all the numeric and one-hot encoded categorical features, alongside the original `churn_flag` label. This "vectorized" format is the standard input for MLlib's machine learning algorithms.

**Output (`outputs/task1/task1_preprocessing_summary.txt`):**
```bash
Task 1: Data Preprocessing and Feature Engineering
Sample Output (after preprocessing):
                                                                                           features  churn_flag
(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 45.0, 19.99, 57.0, 5.55, 160.77, 25.0, 0.0, 5.0, 0.0)           0
    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 68.0, 8.99, 5.0, 1.75, 21.55, 5.0, 5.0, 1.0, 1.0]           1
 (0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 22.0, 8.99, 35.0, 3.61, 164.71, 19.0, 1.0, 4.0, 0.0)           0
```

---

### Task 2: Baseline Model and Detailed Evaluation
**Objective:** Train a baseline Logistic Regression model and conduct a deep-dive evaluation of its performance.

**Explanation of Output:**
The baseline Logistic Regression model achieved perfect scores across all metrics (AUC, Accuracy, Precision, Recall, F1-Score). While this is impressive, a perfect score on a test set often suggests the dataset may be easily separable or could indicate data leakage. The feature importance table is particularly insightful, showing that `plan_type_Premium`and a low `satisfaction_score` are the strongest predictors of churn.

**Output (`outputs/task2/task2_logistic_regression_results.txt`):**
```bash
Task 2: Logistic Regression - Detailed Evaluation

--- Evaluation Metrics ---
AUC: 1.0000
Accuracy: 1.0000
F1-Score: 1.0000

--- Feature Importance (Coefficients) ---
                 feature  coefficient
   plan_type_Vec_Premium     8.328843
      satisfaction_score    -7.036790
  plan_type_Vec_Standard    -4.768374
        region_Vec_North     4.713330
     num_support_tickets     2.728355
```

---

### Task 3: Feature Selection
**Objective:** Identify the most predictive features in the dataset using a statistical test.

**Explanation of Output:**
The Chi-Square test identifies the features that have the strongest statistical relationship with the churn label. The results confirm the importance of subscription details (`plan_type_Vec_Basic`, `monthly_fee`), customer loyalty (`tenure_months`), engagement (`content_watched_per_week`), and service experience (`num_support_tickets`).

**Output (`outputs/task3/task3_feature_selection.txt`):**
```bash
Task 3: Feature Selection using Chi-Square
Top 5 features selected (from indices):
- Index 5: plan_type_Vec_Basic
- Index 10: monthly_fee
- Index 11: tenure_months
- Index 14: content_watched_per_week
- Index 15: num_support_tickets
```

---

### Task 4: Hyperparameter Tuning and Model Comparison
**Objective:** Optimize multiple classification models and compare their performance.

**Explanation of Output:**
This task compares four different models after tuning their key hyperparameters. The results show that Logistic Regression, Decision Tree, and GBT all achieved perfect scores on the test set. The Random Forest model performed slightly lower but still exceptionally well. This indicates that for this particular dataset, multiple models can effectively capture the patterns leading to churn. The GBT model is often a strong candidate due to its robustness.

**Output (`outputs/task4/task4_model_comparison.txt`):**
```bash
Task 4: Hyperparameter Tuning and Model Comparison

Tuning LogisticRegression...
LogisticRegression Best Model - AUC: 1.0000, Accuracy: 1.0000, F1-Score: 1.0000
Best Params for LogisticRegression: {'regParam': 0.0}

Tuning DecisionTree...
DecisionTree Best Model - AUC: 1.0000, Accuracy: 1.0000, F1-Score: 1.0000
Best Params for DecisionTree: {'maxDepth': 5}

Tuning RandomForest...
RandomForest Best Model - AUC: 1.0000, Accuracy: 0.9697, F1-Score: 0.9691
Best Params for RandomForest: {'numTrees': 20}

Tuning GBT...
GBT Best Model - AUC: 1.0000, Accuracy: 1.0000, F1-Score: 1.0000
Best Params for GBT: {'maxIter': 20}
```

---

### Task 5: Business Takeaway
**Objective:** Translate model findings into actionable business insights.

**Explanation of Output:**
This final step synthesizes all the technical findings into strategic recommendations. The key takeaway is that the models can reliably predict churn, allowing the business to move from a reactive to a proactive retention strategy. By focusing on the key drivers identified (like support tickets and satisfaction scores), the company can implement targeted campaigns to improve customer loyalty and reduce churn.

**Output (`outputs/task5/task5_business_takeaway.txt`):**
```bash
Task 5: Business Takeaway

--- 1. What Was Learned ---
The dataset was explored, revealing key customer distributions. Features were successfully preprocessed for machine learning. A Logistic Regression model showed that features like tenure and satisfaction_score were highly predictive of churn. Chi-Square feature selection confirmed the statistical relevance of these features.

--- 2. Model Performance ---
Four models were trained, with the GBTClassifier providing the highest performance (perfect AUC and Accuracy). This indicates the final model can reliably predict customer churn with approximately 100.00% accuracy, allowing the business to proactively target at-risk customers with retention offers.
```

---

## ⚙️ Execution Instructions

### 1. Prerequisites
* An environment with **Apache Spark** and **Python** installed.
* Required Python libraries: `pyspark`, `pandas`, `matplotlib`, `seaborn`.
    ```bash
    pip install pyspark pandas matplotlib seaborn
    ```
* The `churn_streaming_data.csv` file must be in the same directory as the notebook.

### 2. Run the Analysis
Execute the Jupyter Notebook `BigDataInclassAssignment_(1).ipynb`. The script will automatically create a main `outputs` directory with subdirectories for each task (`task0`, `task1`, etc.) and save all result files in their respective folders.

---
