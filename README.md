# Credit Risk Prediction using Artificial Neural Networks (ANN)

## 📌 Project Overview

This project predicts whether a loan applicant is a **Good Credit Risk** or a **Bad Credit Risk** using an **Artificial Neural Network (ANN)**.

The project follows a complete machine learning workflow including data preprocessing, exploratory data analysis, feature engineering, model building, evaluation, and model selection. Multiple ANN architectures were trained and compared to identify the best-performing model.

---

## 📂 Dataset

**Dataset:** German Credit Risk Dataset

The dataset contains information about loan applicants such as:

- Age
- Sex
- Job
- Housing
- Saving Account
- Checking Account
- Credit Amount
- Loan Duration
- Purpose
- Risk (Target Variable)

### Target Variable

| Value | Meaning |
|--------|---------|
| 0 | Good Credit |
| 1 | Bad Credit |

---

## 🎯 Project Objective

The objective of this project is to build an ANN model capable of predicting whether a customer is a good or bad credit risk based on their financial and demographic information.

The final model can assist financial institutions in making informed loan approval decisions.

---

# 🛠 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- TensorFlow / Keras
- Joblib
- Jupyter Notebook


---

# 🔄 Machine Learning Pipeline

### 1. Data Collection

- German Credit Dataset

### 2. Data Cleaning

- Missing value treatment
- Duplicate checking
- Data type verification

### 3. Exploratory Data Analysis

- Distribution plots
- Count plots
- Feature analysis
- Class distribution

### 4. Feature Engineering

- Label Encoding
- One-Hot Encoding
- Feature Selection

### 5. Data Preprocessing

- Train-Test Split
- Feature Scaling using StandardScaler

### 6. ANN Model Building

Seven different ANN architectures were designed and compared.

The models differed in:

- Number of hidden layers
- Number of neurons
- Dropout
- Optimizer
- Learning strategy

### 7. Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score

Each model was trained **three times**, and the average performance was used for comparison to reduce the effect of random initialization.

---

# 🧠 Artificial Neural Network

The final model consists of:

- Input Layer
- Hidden Layer(s)
- ReLU Activation
- Dropout Layer(s)
- Output Layer
- Sigmoid Activation

Loss Function:

```
Binary Crossentropy
```

Optimizer:

```
Adam
```

---

# 📊 Model Comparison

Seven ANN models were trained.

The final model was selected based on:

- Highest F1 Score
- Good Precision
- Good Recall
- Highest Overall Accuracy

Final Selected Model:

> **Model 5**

---

# 📈 Final Model Performance

| Metric | Score |
|---------|--------|
| Accuracy | **76.50%** |
| Precision | **67.36%** |
| Recall | **42.22%** |
| F1 Score | **51.75%** |

---

# 📉 Evaluation Metrics

The final ANN model was evaluated using:

- Classification Report
- Confusion Matrix
- ROC Curve
- ROC-AUC Score
- Precision-Recall Curve
- Average Precision (AP)

### ROC-AUC Score

```
0.76
```

The ROC curve indicates that the model has good discriminative capability in distinguishing between good-credit and bad-credit customers.

### Precision-Recall Curve

Average Precision (AP):

```
0.58
```

The Precision-Recall curve demonstrates a reasonable balance between identifying bad-credit customers and minimizing false alarms.

---

# 💾 Saved Model

The trained ANN model is saved as:

```
credit_risk_ann.keras
```

The fitted scaler is saved as:

```
scaler.pkl
```

These files allow future predictions without retraining the model.


---

# 📊 Results

Among seven ANN architectures, **Model 5** achieved the best overall performance.

The model provides a balanced trade-off between:

- Accuracy
- Precision
- Recall

It successfully identifies a significant proportion of bad-credit customers while maintaining relatively few false positive predictions.



---

# 📚 Learning Outcomes

This project helped in understanding:

- Data preprocessing
- Feature engineering
- Artificial Neural Networks
- Binary classification
- Model evaluation
- ROC & Precision-Recall Curves
- Model comparison
- Model serialization
- End-to-end machine learning workflow

---

# 👨‍💻 Author

**Ashutosh Kashyap**

BS-MS in Artificial Intelligence & Cyber Security

Indian Institute of Technology Patna

---
