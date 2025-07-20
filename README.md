# Diabetes Prediction and Analysis

This project explores a diabetes dataset using data visualization, preprocessing, and logistic regression modeling. The goal is to predict diabetes outcomes based on medical attributes.

---

## 📊 Dataset Overview

The dataset contains the following columns:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (Target: 1 for diabetic, 0 for non-diabetic)

---

## 🔍 Exploratory Data Analysis (EDA)

### a. Count of Outcome Classes

```python
sns.countplot(x='Outcome', data=df)
plt.title("Count of Diabetes Outcome")
plt.savefig("images/countplot_outcome.png")
plt.show()
```

📷 ![Countplot](images/countplot_outcome.png)

---

### b. Correlation Heatmap

```python
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.savefig("images/correlation_heatmap.png")
plt.show()
```

📷 ![Heatmap](images/correlation_heatmap.png)

---

### c. Glucose & Other Feature Distributions

```python
for col in df.columns[:-1]:
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.savefig(f"images/histplot_{col}.png")
    plt.show()
```

📷 ![Histogram Glucose](images/histplot_Glucose.png)

---

## 🧹 Data Preprocessing

- **Replaced 0s** with `NaN` for features where zero is not physiologically valid: `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`.
- **Imputed** missing values with median of each feature.
- **Split** dataset into training and testing sets.

---

## 🤖 Model Training

Used **Logistic Regression** from `sklearn.linear_model`.  
Split data: 75% training, 25% testing.

**Before cleaning:** Accuracy = **74.68%**  
**After cleaning:** Accuracy = **75.32%** ✅

### Classification Report:

```
              precision    recall  f1-score   support

           0       0.80      0.83      0.81        99
           1       0.67      0.62      0.64        55

    accuracy                           0.75       154
   macro avg       0.73      0.72      0.73       154
weighted avg       0.75      0.75      0.75       154
```

### Confusion Matrix:

```
[[82 17]
 [21 34]]
```

---

## 📌 Insights and Conclusions

### 🧠 Who is more vulnerable?
- People with **higher glucose levels**, **BMI**, and **age** are more likely to have diabetes.
- Females with **multiple pregnancies** are also at higher risk.
- The model reveals that **Glucose** is the most influential factor, followed by **BMI** and **Age**.

### ⚠️ Where do the outliers come from?
- Features like **Insulin**, **SkinThickness**, and **BloodPressure** have many **zero values**, which are likely **missing values** entered incorrectly.
- These were replaced with `NaN` and imputed to improve the model.

### 🔗 Are there relationships between the factors?
- Strong positive correlation between **Glucose** and **Outcome**.
- Moderate relationship between **BMI** and **Outcome**.
- Weak or no correlation between **SkinThickness** and **Outcome**, suggesting it may not be a strong predictor.

---

## 📁 Folder Structure

```
project-folder/
│
├── images/
│   ├── countplot_outcome.png
│   ├── correlation_heatmap.png
│   ├── histplot_Pregnancies.png
│   ├── histplot_BloodPressure.png
│   ├── histplot_BMI.png
│   ├── histplot_DiabetesPedigreeFunction.png
│   ├── histplot_Insulin.png
│   ├── histplot_SkinThickness.png
│   ├── histplot_Glucose.png
│   ├── histplot_Age.png
│   ├── countplot_outcome_after_cleaning.png
│   ├── correlation_heatmap_after_cleaning.png
│   ├── histplot_Pregnancies_after_cleaning.png
│   ├── histplot_BloodPressure_after_cleaning.png
│   ├── histplot_BMI_after_cleaning.png
│   ├── histplot_DiabetesPedigreeFunction_after_cleaning.png
│   ├── histplot_Insulin_after_cleaning.png
│   ├── histplot_SkinThickness_after_cleaning.png
│   ├── histplot_Glucose_after_cleaninge.png
│   └── histplot_Age_after_cleaning.png
│
├── diabetes.csv
├── Diabetes Prediction and Analysis.ipynb
└── README.md
```

---

## ✅ Next Steps

- Try different classifiers (Random Forest, XGBoost).
- Address class imbalance using SMOTE or resampling.
- Tune hyperparameters for better performance.
