# 📊 YouTube Shorts Success Prediction

## 📌 Project Overview

This project predicts whether a **YouTube Short will be successful** using **machine learning**, focusing strictly on **realistic, pre-upload predictions**.

A full end-to-end data science workflow was implemented:

* Exploratory Data Analysis (EDA)
* Feature engineering
* Multiple classification models
* Stratified train/test split
* Model evaluation
* Deployment with **Streamlit**

---

## 🎯 Problem Statement

Content creators want to estimate *before publishing* whether a Short has a high chance of success.

**Objective:**

> Predict if a YouTube Short will be successful using only metadata available before upload.

---

## 🧠 Target Variable

* **`is_successful`** (binary classification)

  * `1` → Successful Short
  * `0` → Not successful

---

## 📂 Dataset

**Source:** Kaggle – YouTube Shorts Performance Dataset

### Available Features

* `duration_sec`
* `hashtags_count`
* `views`
* `likes`
* `comments`
* `shares`
* `upload_hour`
* `category`

---

## ⚠️ Real-World Modeling Decision

Engagement variables such as `views`, `likes`, `comments`, and `shares` are **not available before publishing**.

Therefore:

* ✅ Used for **EDA and insights**
* ❌ Excluded from the **prediction model**

This prevents **data leakage** and ensures the model can be realistically deployed.

---

## 🛠️ Feature Engineering

* `is_successful` new target created from views > quatile .75
* `category` encoded numerically

---

## 🔀 Train/Test Split

The dataset is **imbalanced**, with fewer successful Shorts.

To ensure reliable evaluation, a **stratified split** was used:

```python
train_test_split(X, y, test_size=0.2, stratify=y)
```

This preserves class distribution in both training and test sets.

---

## 🤖 Models Trained

* Gradient Boosting (Best metrics)

---

## 📊 Evaluation Metrics

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

Due to class imbalance, **Accuracy alone is not sufficient**.

---

## 🎚️ Decision Threshold

The model uses the **default classification threshold (0.5)** provided by scikit-learn.

### Rationale

After re-training using **stratified sampling**, the metrics became stable and reliable. Given the scope of the project, no additional threshold tuning was required.

---

## 🚀 Streamlit App

A professional **Streamlit application** was built for interactive predictions.

### App Characteristics

* Uses **only pre-upload features**
* Displays **probability of success**
* Applies a clear and interpretable decision rule
* Clean, business-ready UI

### Model Inputs

* `duration_sec`
* `hashtags_count`
* `upload_hour`
* `category`

### Output

* Probability of success
* Final classification (Successful / Not Successful)

---


## 🧩 Tech Stack

* Python
* Pandas / NumPy
* Scikit-learn
* Streamlit
* Matplotlib / Seaborn

---

## 💡 Key Learnings

* Stratified sampling is critical for imbalanced classification
* Accuracy is misleading when used alone
* Preventing data leakage is essential for deployable models
* Simple, interpretable solutions are often preferred


---

## Author

**Isaí Cruz**

Data Science | Machine Learning | Analytics
