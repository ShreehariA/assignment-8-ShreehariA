# **Ensemble Learning for Complex Regression Modeling on Bike Share Data (DA5401 A8)**

**Student Name:** Shreehari Anbazhagan\
**Roll Number:** DA25C020

---

## **Project Overview**

This project focuses on ensemble learning techniques for regression modeling using the Bike Share dataset. The primary objective is to compare different ensemble methods (Bagging, Boosting, and Stacking) and demonstrate their effectiveness in reducing bias and variance compared to baseline models.

The analysis evaluates five distinct models, including baseline regressors (Linear Regression and Decision Tree) and advanced ensemble techniques (Bagging, Gradient Boosting, and Stacking). The core of the project is to use **Root Mean Squared Error (RMSE)** as the evaluation metric and demonstrate how ensemble methods can achieve superior performance by balancing the bias–variance trade-off.

**Dataset:** Bike Share Hourly Data (`hour.csv`)

---

## Folder Structure & Files

```
project-root/
│
├─ main.ipynb              # Core Jupyter Notebook with all code, analysis, and visualizations
├─ datasets/               # Dataset files
│  ├─ hour.csv             # Hourly bike share data
│  └─ day.csv              # Daily bike share data (optional)
├─ instructions/           # Assignment PDF and reference materials
│  └─ A8 Ensembles.pdf     # Assignment instructions
├─ pyproject.toml          # Project dependencies
├─ uv.lock                 # Locked dependency versions (for uv sync)
└─ README.md               # This project summary
```

### Notes:

*   Run `uv sync` to install dependencies exactly as tested.
*   The dataset should be placed in the `datasets/` folder.
*   The notebook is **self-contained**: all data loading, preprocessing, model training, and evaluation steps are reproducible.

---

## Dependencies

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
```

---

## **Analysis Workflow**

The project is structured into a comprehensive ensemble learning pipeline:

1.  **Part A: Data Preprocessing and Baseline**
    *   Load and explore the Bike Share hourly dataset
    *   Drop irrelevant columns (`instant`, `dteday`, `casual`, `registered`)
    *   Perform feature engineering: one-hot encoding for categorical features (`season`, `weathersit`, `mnth`, `hr`, `weekday`)
    *   Split data into training (80%) and testing (20%) sets
    *   Train baseline models: **Linear Regression** (with preprocessing pipeline) and **Decision Tree** (max_depth=6)
    *   Evaluate baseline performance using RMSE

2.  **Part B: Ensemble Techniques for Bias and Variance Reduction**
    *   **Bagging (Variance Reduction)**: Train a `BaggingRegressor` with 50 Decision Tree estimators to reduce variance
    *   **Boosting (Bias Reduction)**: Train a `GradientBoostingRegressor` with 100 estimators (learning_rate=0.1, max_depth=6) to reduce bias
    *   Compare ensemble methods with baseline models

3.  **Part C: Stacking for Optimal Performance**
    *   Build a **Stacking Regressor** with three base learners:
        *   K-Nearest Neighbors Regressor
        *   Bagging Regressor (Decision Trees)
        *   Gradient Boosting Regressor
    *   Use **Ridge Regression** as the meta-learner to combine base model predictions
    *   Evaluate the stacking ensemble's performance

4.  **Part D: Final Analysis**
    *   Create a comprehensive comparison table of all models
    *   Generate visualizations (bar charts) comparing RMSE across all models
    *   Provide interpretation and conclusions about ensemble learning effectiveness

---

## **Key Findings**

The analysis revealed a clear hierarchy of model performance, with advanced ensemble methods significantly outperforming baseline models.

*   **Stacking Dominates**: The **Stacking Regressor** achieved the lowest RMSE (40.07), demonstrating the power of combining diverse models through a meta-learner.
*   **Boosting Shows Strong Performance**: **Gradient Boosting Regressor** achieved an RMSE of 41.68, showing excellent bias reduction capabilities.
*   **Bagging Reduces Variance**: **Bagging Regressor** achieved an RMSE of 96.02, improving upon the single Decision Tree (99.02) by reducing variance.
*   **Baseline Comparison**: Both baseline models (Linear Regression: 100.45, Decision Tree: 99.02) were outperformed by all ensemble methods.

| Model | RMSE |
| :--- | :---: |
| **Stacking Regressor** | **40.07** |
| Gradient Boosting Regressor | 41.68 |
| Bagging Regressor | 96.02 |
| Decision Tree (Single) | 99.02 |
| Linear Regression | 100.45 |

---

## **Conclusion and Recommendations**

For this bike share demand prediction task, the **Stacking Regressor** is the most highly recommended model.

This recommendation is justified by the following:

1.  **Superior Performance**: It achieved the lowest RMSE (40.07), outperforming all other models by a significant margin. This represents a 60% improvement over the baseline Linear Regression model.

2.  **Effective Bias–Variance Trade-off**: Stacking combines multiple diverse models—each with different learning biases—into a single meta-model that learns to optimally weight their predictions. While Bagging primarily reduces variance and Boosting focuses on reducing bias, Stacking leverages **model diversity** to capture both linear and nonlinear relationships.

3.  **Robust Ensemble Architecture**: The Ridge meta-learner integrates the strengths of each base model (KNN, Bagging, and Boosting), compensating for their individual weaknesses. This results in more stable and accurate predictions than any single model.

4.  **Practical Benefits**: The Stacking Regressor achieves a more stable and accurate prediction than any single model, effectively minimizing overall generalization error and exemplifying the **bias–variance trade-off** principle in practice.

While Gradient Boosting is a very strong alternative (RMSE: 41.68), Stacking's superior performance and ability to combine diverse models make it the definitive choice for this regression problem.

---

## **Key Concepts Demonstrated**

*   **Bias–Variance Trade-off**: Understanding how different ensemble methods address bias and variance
*   **Bagging (Bootstrap Aggregating)**: Variance reduction through averaging multiple models trained on bootstrap samples
*   **Boosting**: Bias reduction through sequential model training that focuses on previous errors
*   **Stacking**: Combining diverse base models using a meta-learner for optimal performance
*   **Feature Engineering**: Proper handling of categorical variables through one-hot encoding
*   **Model Evaluation**: Using RMSE as an appropriate metric for regression problems