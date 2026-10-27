# Credit Risk Prediction

An end-to-end machine learning project that predicts whether a loan applicant represents a **Good Credit Risk** or a **Bad Credit Risk**. The project covers exploratory data analysis, data preprocessing, model comparison, hyperparameter tuning, ensemble learning, model evaluation, REST API development, and cloud deployment.

**Live Application:** [Credit Risk Prediction — Streamlit](https://credit-risk-prediction04.streamlit.app/)

**Backend API:** [FastAPI on Render](https://credit-risk-prediction-nkak.onrender.com)

**GitHub Repository:** [Credit-risk-prediction](https://github.com/ashutoshkashyap04/Credit-risk-prediction)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Project Objectives](#project-objectives)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Machine Learning Workflow](#machine-learning-workflow)
  - [1. Data Loading and Exploratory Data Analysis](#1-data-loading-and-exploratory-data-analysis)
  - [2. Data Preprocessing](#2-data-preprocessing)
  - [3. Model Training and Evaluation](#3-model-training-and-evaluation)
  - [4. Hyperparameter Tuning](#4-hyperparameter-tuning)
  - [5. Stacking Classifier](#5-stacking-classifier)
  - [6. Final Model Evaluation](#6-final-model-evaluation)
- [Model Performance](#model-performance)
- [Application Architecture](#application-architecture)
- [Deployment](#deployment)
- [Run Locally](#run-locally)
- [API Usage](#api-usage)
- [Key Learnings](#key-learnings)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Author](#author)

---

## Project Overview

Credit risk assessment is an important task in the financial services industry. Lenders need to assess the likelihood that applicants will repay their loans to help manage potential credit losses.

This project develops a binary classification model using the German Credit Risk Dataset. It uses applicant demographic information, account balances, credit amount, loan duration, and loan purpose to classify applicants into good-credit and bad-credit categories.

Rather than relying on accuracy alone, the project compares multiple evaluation metrics, including precision, recall, F1-score, and ROC-AUC. This is particularly important because the dataset contains more good-credit applicants than bad-credit applicants.

After comparing nine machine learning algorithms and tuning the strongest candidates, a **Stacking Classifier** was selected as the final model. The trained model was integrated into a FastAPI backend and a Streamlit frontend, and both components were deployed online.

## Problem Statement

Financial institutions face the challenge of distinguishing applicants who represent relatively low credit risk from those who may have difficulty repaying their loans.

A model that predicts every applicant as good credit could achieve deceptively high accuracy while failing to identify risky applicants. Therefore, credit risk prediction requires a balanced evaluation of correctly identifying bad-credit applicants and avoiding incorrect risk classifications.

The objective is to build and deploy a machine learning application that predicts credit risk from applicant information and demonstrates an end-to-end data science workflow.

## Project Objectives

- Explore and understand the distribution of applicant characteristics and credit risk.
- Identify and handle missing values in savings and checking account information.
- Encode categorical variables into numerical features suitable for machine learning.
- Preserve the target class distribution using stratified train-test splitting.
- Train and compare nine baseline machine learning algorithms.
- Select promising models using the F1-score.
- Optimize model hyperparameters using cross-validation.
- Compare tuned models with a Stacking Classifier.
- Evaluate the selected model using classification metrics and a confusion matrix.
- Save the trained model and scaler for inference.
- Develop a REST API using FastAPI.
- Build an interactive prediction interface using Streamlit.
- Deploy the backend on Render and the frontend on Streamlit Community Cloud.

## Dataset Description

The project uses the **German Credit Risk Dataset**, containing 1,000 loan applicant records and 10 columns, including the target variable.

### Dataset overview

| Property | Description |
|---|---|
| Total records | 1,000 |
| Original columns | 10 |
| Training records | 800 |
| Testing records | 200 |
| Problem type | Binary classification |
| Target variable | `Risk` |
| Good-credit applicants | 700 (70%) |
| Bad-credit applicants | 300 (30%) |

### Features

| Feature | Description |
|---|---|
| `Age` | Age of the applicant |
| `Sex` | Applicant's recorded sex |
| `Job` | Encoded job skill level |
| `Housing` | Housing status |
| `Saving accounts` | Savings account category |
| `Checking account` | Checking account category |
| `Credit amount` | Amount of credit requested |
| `Duration` | Credit duration |
| `Purpose` | Purpose of the credit |
| `Risk` | Target classification |

### Target variable

The target variable `Risk` is encoded as follows:

| Encoded value | Meaning |
|---|---|
| `0` | Good Credit — lower-risk class |
| `1` | Bad Credit — higher-risk class |

The dataset contains a **70:30 class distribution**, making it moderately imbalanced. Consequently, accuracy alone is insufficient for evaluating the model.

## Project Structure

The repository is organized into separate directories for notebooks, datasets, model artifacts, the backend API, and the frontend application.

```text
Credit-risk-prediction/
│
├── backend/
│   ├── app.py
│   └── schemas.py
│
├── data/
│   ├── raw/
│   │   └── cleaned_data.csv
│   └── processed/
│       ├── train.csv
│       └── test.csv
│
├── frontend/
│   └── main.py
│
├── models/
│   ├── stacking_classifier_model.pkl
│   └── scaler.pkl
│
├── notebooks/
│   ├── 01_data_loading_and_eda.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_model_training_and_evaluation.ipynb
│
├── .gitignore
├── requirements.txt
└── README.md
```

*The tree documents the project files and artifacts used in the workflow; generated datasets and model files may be excluded from version control depending on the repository's current contents.*

### Directory description

- **`notebooks/`** — Contains the three notebooks for EDA, preprocessing, model training, tuning, and evaluation.
- **`data/raw/`** — Contains the cleaned dataset generated during exploratory analysis.
- **`data/processed/`** — Contains the training and testing datasets used for model development.
- **`models/`** — Stores the fitted Stacking Classifier and the scaler used during training.
- **`backend/`** — Contains FastAPI application logic and Pydantic request validation.
- **`frontend/`** — Contains the Streamlit user interface.
- **`requirements.txt`** — Lists the Python dependencies.
- **`.gitignore`** — Specifies files and directories excluded from version control.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming language | Python |
| Data manipulation | Pandas, NumPy |
| Data visualization | Matplotlib, Seaborn |
| Machine learning | Scikit-learn, XGBoost |
| Preprocessing | Pandas One-Hot Encoding, StandardScaler |
| Model evaluation | Classification metrics, confusion matrix, ROC-AUC |
| Hyperparameter optimization | RandomizedSearchCV |
| Ensemble learning | StackingClassifier |
| Model serialization | Joblib |
| Backend | FastAPI, Pydantic |
| Frontend | Streamlit |
| API communication | REST API, HTTP requests |
| Deployment | Render, Streamlit Community Cloud |
| Development environment | Jupyter Notebook, VS Code |

---

## Machine Learning Workflow

### 1. Data Loading and Exploratory Data Analysis

**Notebook:** `01_data_loading_and_eda.ipynb`

The first notebook focuses on understanding the dataset, examining data quality, and identifying patterns in applicant characteristics.

#### Initial data inspection

The original dataset contains 1,000 rows and 10 columns. The initial inspection includes:

- Dataset dimensions and column names.
- Data types and non-null counts.
- Missing values in individual columns.
- Distribution of the target variable.
- Distribution of categorical and numerical features.
- Checks for invalid values in numerical columns.

#### Missing value treatment

Missing values were present in two account-related columns:

| Column | Missing values |
|---|---:|
| `Saving accounts` | 183 |
| `Checking account` | 394 |

Instead of dropping these records, missing account information was replaced with the category `No Account`.

This preserves the dataset's original 1,000 records and allows the absence of account information to be represented explicitly.

#### Job-level mapping

The `Job` feature originally contains integer codes. These were mapped to descriptive labels for easier interpretation during visualization.

```python
job_mapping = {
    0: "Unskilled_Non-resident",
    1: "Unskilled_Resident",
    2: "Skilled",
    3: "Manager"
}
```

#### Data validation

The notebook also checks for invalid negative values in numerical features such as age, credit amount, and duration.

#### Exploratory visualizations

Count plots and other relevant visualizations were used to investigate:

- Gender distribution.
- Housing categories.
- Job-level distribution.
- Savings and checking account categories.
- Credit purposes.
- Credit risk class distribution.
- Distributions of numerical features.

The cleaned dataset was saved to:

```text
data/raw/cleaned_data.csv
```

### 2. Data Preprocessing

**Notebook:** `02_data_preprocessing.ipynb`

The second notebook prepares the cleaned data for machine learning.

#### One-hot encoding

One-hot encoding was applied to the following categorical features:

- `Sex`
- `Housing`
- `Purpose`

Using `pandas.get_dummies()` converts categorical values into numerical indicator columns.

#### Ordinal mapping of account categories

Savings account categories were mapped to integer values:

```python
saving_map = {
    "No Account": 0,
    "little": 1,
    "moderate": 2,
    "quite rich": 3,
    "rich": 4
}
```

Checking account categories were mapped as follows:

```python
checking_map = {
    "No Account": 0,
    "little": 1,
    "moderate": 3,
    "rich": 4
}
```

These mappings make the account features suitable for numerical model training. They also impose an ordering on the encoded categories, which is an assumption of this preprocessing approach.

#### Target encoding

The target variable was encoded as:

```text
bad  → 1
good → 0
```

This makes the problem a binary classification task in which class `1` represents the higher-risk category.

#### Train-test split

The dataset was divided into training and testing sets using an 80:20 split with stratification.

| Dataset | Shape |
|---|---|
| Training dataset | `(800, 17)` |
| Testing dataset | `(200, 17)` |

Stratification preserves approximately the original target class proportions in both subsets.

The resulting datasets were saved to:

```text
data/processed/train.csv
data/processed/test.csv
```

### 3. Model Training and Evaluation

**Notebook:** `03_model_training_and_evaluation.ipynb`

The third notebook contains model training, baseline comparison, hyperparameter tuning, ensemble construction, and final evaluation.

The training and testing datasets were separated into features and target labels:

- `X_train`
- `X_test`
- `y_train`
- `y_test`

`StandardScaler` was used to scale the training features. The fitted scaler was retained for subsequent use when preparing inputs for the deployed model.

#### Baseline model comparison

Nine machine learning algorithms were trained and evaluated.

| No. | Model |
|---:|---|
| 1 | Logistic Regression |
| 2 | K-Nearest Neighbors (KNN) |
| 3 | Gaussian Naive Bayes |
| 4 | Decision Tree |
| 5 | Random Forest |
| 6 | Support Vector Machine (SVM) |
| 7 | AdaBoost |
| 8 | Gradient Boosting |
| 9 | XGBoost |

#### Evaluation metrics

The models were evaluated using:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted bad-credit applicants who actually belong to the bad-credit class.
- **Recall:** Proportion of actual bad-credit applicants correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures the model's ability to distinguish between the two classes across classification thresholds.

Since false negatives can be costly in credit risk assessment, recall is important. Precision is also important because excessive false positives can result in good-credit applicants being incorrectly classified as risky.

The F1-score provides a useful balance between these two metrics.

#### Baseline results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.760 | 0.620 | 0.517 | 0.564 | 0.759 |
| Decision Tree | 0.720 | 0.534 | 0.517 | 0.525 | 0.662 |
| Random Forest | 0.755 | 0.634 | 0.433 | 0.515 | 0.762 |
| Gradient Boosting | 0.740 | 0.587 | 0.450 | 0.509 | 0.766 |
| Gaussian Naive Bayes | 0.665 | 0.439 | 0.417 | 0.427 | 0.654 |
| AdaBoost | 0.720 | 0.556 | 0.333 | 0.417 | 0.769 |
| KNN | 0.700 | 0.500 | 0.283 | 0.362 | 0.664 |
| Logistic Regression | 0.725 | 0.609 | 0.233 | 0.337 | 0.647 |
| SVM | 0.720 | 0.700 | 0.117 | 0.200 | 0.706 |

*Values are rounded to three decimal places. The F1-score, precision, and recall refer to the bad-credit class (class 1).*

The six strongest candidates based on F1-score were selected for further optimization.

### 4. Hyperparameter Tuning

The six selected candidates were subjected to hyperparameter tuning using `RandomizedSearchCV` with five-fold cross-validation and F1-score as the optimization metric.

This process searches for promising combinations of model parameters rather than relying entirely on default configurations.

The best-performing tuned candidates were:

| Model | Best cross-validation F1-score |
|---|---:|
| Decision Tree | 0.5442 |
| XGBoost | 0.5436 |
| Gradient Boosting | 0.5042 |

The best hyperparameter combinations obtained during tuning were used to construct the final candidate estimators.

### 5. Stacking Classifier

After tuning, the three strongest candidates were compared individually and as an ensemble.

A `StackingClassifier` was constructed using:

- **Base estimators:** Tuned Decision Tree, XGBoost, and Gradient Boosting classifiers.
- **Final estimator:** Logistic Regression.

Stacking is an ensemble learning technique in which a meta-model learns how to combine predictions from multiple base models.

The objective was to leverage the different predictive patterns captured by the individual algorithms rather than rely on a single classifier.

#### Final candidate comparison

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.775 | 0.636 | 0.583 | 0.609 | 0.765 |
| Stacking Classifier | 0.785 | 0.698 | 0.500 | 0.583 | 0.782 |
| XGBoost | 0.760 | 0.625 | 0.500 | 0.556 | 0.784 |
| Decision Tree | 0.705 | 0.533 | 0.133 | 0.213 | 0.720 |

The Stacking Classifier was selected as the final model because it achieved the highest accuracy among the final candidates, the highest precision, and a strong ROC-AUC score.

Gradient Boosting achieved a higher F1-score and recall, so stacking is not the best model on every metric. The final selection reflects a trade-off among classification performance measures.

### 6. Final Model Evaluation

The selected Stacking Classifier was evaluated on the held-out test dataset containing 200 applicants.

#### Confusion matrix

```text
[[127, 13],
 [ 30, 30]]
```

The confusion matrix can be interpreted as follows:

| Actual class | Predicted good credit (0) | Predicted bad credit (1) |
|---|---:|---:|
| Good credit (0) | 127 | 13 |
| Bad credit (1) | 30 | 30 |

- **True negatives (127):** Good-credit applicants correctly classified as good credit.
- **False positives (13):** Good-credit applicants incorrectly classified as bad credit.
- **False negatives (30):** Bad-credit applicants incorrectly classified as good credit.
- **True positives (30):** Bad-credit applicants correctly classified as bad credit.

The model correctly classified 157 of the 200 test applicants.

#### Classification report

```text
              precision    recall  f1-score   support

           0       0.81      0.91      0.86       140
           1       0.70      0.50      0.58        60

    accuracy                           0.79       200
   macro avg       0.75      0.70      0.72       200
weighted avg       0.78      0.79      0.77       200
```

The final model achieved:

- **Accuracy:** 78.5%
- **Precision for bad credit:** Approximately 69.8%
- **Recall for bad credit:** 50.0%
- **F1-score for bad credit:** Approximately 58.3%
- **ROC-AUC:** 0.782

The model identifies half of the actual bad-credit applicants in the test set. The remaining half are false negatives, highlighting an important area for future improvement.

The fitted model was serialized using Joblib:

```python
import joblib

joblib.dump(
    stack_model,
    "../models/stacking_classifier_model.pkl"
)
```

The fitted scaler was also saved for use during inference.

---

## Model Performance

The following chart summarizes the final model's test-set metrics.

| Metric | Score |
|---|---:|
| Accuracy | 78.5% |
| Precision — bad-credit class | 69.8% |
| Recall — bad-credit class | 50.0% |
| F1-score — bad-credit class | 58.3% |
| ROC-AUC | 78.2% |

These metrics should be interpreted together. Accuracy measures overall correctness, while recall measures how many risky applicants are identified. Precision measures how often a bad-credit prediction is correct, and ROC-AUC measures ranking discrimination across thresholds.

**Important:** These are test-set results from a small dataset of 200 applicants. They do not establish that the model is suitable for real lending decisions. Further validation, fairness analysis, probability calibration, and threshold selection would be necessary before practical financial use.

## Application Architecture

The deployed application separates the user interface from the machine learning inference service.

```text
User
  |
  v
Streamlit Frontend
  |
  | Applicant information
  v
HTTP POST Request
  |
  v
FastAPI Backend
  |
  | Validate input using Pydantic
  | Apply the saved preprocessing/scaling
  | Load the trained classifier
  v
Stacking Classifier
  |
  | Predict credit risk
  v
API JSON Response
  |
  v
Streamlit Prediction Result
```

### FastAPI backend

The backend is implemented in `backend/app.py`, with input validation defined in `backend/schemas.py`.

Its responsibilities include:

- Accepting applicant information through an HTTP endpoint.
- Validating incoming data.
- Preparing the input in the expected feature format.
- Applying the saved scaler.
- Generating a prediction using the trained Stacking Classifier.
- Returning the prediction to the frontend.

The backend is deployed on Render.

### Streamlit frontend

The frontend is implemented in `frontend/main.py`.

Its responsibilities include:

- Collecting applicant details through an interactive interface.
- Sending the input to the deployed FastAPI endpoint.
- Receiving the API response.
- Displaying the predicted credit risk in a user-friendly format.

The frontend is deployed on Streamlit Community Cloud.

## Deployment

Both components are available online.

| Component | Platform | Link |
|---|---|---|
| Frontend | Streamlit Community Cloud | [Open application](https://credit-risk-prediction04.streamlit.app/) |
| Backend | Render | [Open API service](https://credit-risk-prediction-nkak.onrender.com) |

The frontend communicates with the backend using the deployed API endpoint:

```text
https://credit-risk-prediction-nkak.onrender.com/predict
```

The deployed frontend and backend are separate services, allowing the prediction interface to communicate with the model through a REST API.

## Run Locally

Follow these steps to run the project on your local machine.

### 1. Clone the repository

```bash
git clone https://github.com/ashutoshkashyap04/Credit-risk-prediction.git

cd Credit-risk-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows using PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or use Git Bash:

```bash
source .venv/Scripts/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI backend

From the project root, run:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

The API should now be available at:

```text
http://127.0.0.1:8000
```

FastAPI's interactive documentation is normally available at:

```text
http://127.0.0.1:8000/docs
```

### 5. Start the Streamlit frontend

Open a second terminal, activate the same virtual environment, and run:

```bash
streamlit run frontend/main.py
```

The Streamlit interface will display a local URL in the terminal.

**Configuration note:** Ensure that the frontend's API URL points to the backend you want to use. For local testing, use `http://127.0.0.1:8000/predict`; for the deployed application, use the Render endpoint listed in the deployment section.

The saved model and scaler must also be available at the paths expected by the backend.

## API Usage

The FastAPI backend exposes a prediction endpoint:

```text
POST /predict
```

It accepts applicant information and returns a credit risk prediction.

### Example request

```json
{
  "Age": 30,
  "Sex": "male",
  "Job": 2,
  "Housing": "own",
  "Saving_account": "moderate",
  "Checking_account": "moderate",
  "Credit_amount": 2000,
  "Duration": 12,
  "Purpose": "car"
}
```

The field names must match the Pydantic schema defined in `backend/schemas.py`. In particular, `Saving_account` and `Checking_account` use underscores because spaces are not used in the request field names.

### Example response

The API returns the predicted risk classification. Conceptually, the result maps to:

```json
{
  "prediction": 0,
  "risk": "Good Credit"
}
```

This response illustrates the prediction mapping; the exact response keys depend on the implementation in `backend/app.py`.

| Prediction | Classification |
|---|---|
| `0` | Good Credit |
| `1` | Bad Credit |

## Key Learnings

This project provided practical experience in several areas of applied machine learning and deployment:

- Performing exploratory data analysis on financial data.
- Handling missing categorical information without removing records.
- Encoding categorical variables for machine learning.
- Creating stratified training and testing datasets.
- Scaling numerical features for inference.
- Comparing multiple classification algorithms using appropriate metrics.
- Performing hyperparameter optimization with cross-validation.
- Combining models through ensemble learning.
- Interpreting confusion matrices and classification reports.
- Saving trained artifacts with Joblib.
- Developing a REST API with FastAPI and Pydantic.
- Connecting a Streamlit frontend to a remote API.
- Deploying a complete machine learning application using Render and Streamlit Community Cloud.

## Limitations and Future Improvements

Although the project demonstrates a complete machine learning and deployment workflow, several improvements could make it more robust.

### Model performance

- Improve recall for the bad-credit class to reduce missed risky applicants.
- Investigate classification threshold optimization based on the relative costs of false positives and false negatives.
- Evaluate additional models and feature engineering strategies.
- Assess probability calibration and use cross-validation for more reliable performance estimates.

### Data and preprocessing

- Compare the current account-category mappings with alternative encoding approaches.
- Investigate the impact of feature scaling and preprocessing choices.
- Evaluate performance on larger, more representative datasets.
- Review feature quality, data leakage risks, and reproducibility.

### Production readiness

- Add automated tests for input validation, preprocessing, and API predictions.
- Add monitoring for prediction errors and changes in incoming data.
- Improve API error handling and operational logging.
- Conduct fairness and bias assessments before any real-world lending application.

### Business validation

The German Credit Risk Dataset is useful for educational and experimental work, but its size and scope limit conclusions about real financial institutions. Any practical deployment would require appropriate data governance, independent validation, fairness evaluation, and domain-specific risk controls.

## Author

**Ashutosh Kashyap**

BS-MS in Artificial Intelligence & Cyber Security  
Indian Institute of Technology Patna

- **GitHub:** [ashutoshkashyap04](https://github.com/ashutoshkashyap04)
- **Project Repository:** [Credit-risk-prediction](https://github.com/ashutoshkashyap04/Credit-risk-prediction)
- **Live Application:** [Credit Risk Prediction](https://credit-risk-prediction04.streamlit.app/)

---

*This project is intended for educational and demonstration purposes. Its predictions should not be used as the sole basis for real credit approval, rejection, or other consequential financial decisions.*
