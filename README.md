
# Classification Mini-Project: Telco Customer Churn Prediction & Deployment

## Project Overview

This project demonstrates an end-to-end Machine Learning classification workflow to predict customer churn in the telecommunications industry.

The goal is to identify customers who are likely to cancel their subscriptions, helping businesses make data-driven customer retention decisions.

The project covers data exploration, preprocessing, training and comparing seven classification algorithms, hyperparameter tuning, model evaluation, model serialization, and deployment through an interactive Streamlit application.

## Business Problem

Customer churn is a major challenge for telecommunications companies because losing existing customers can affect revenue and increase customer acquisition costs.

By predicting which customers are likely to churn, businesses can identify at-risk customers and prioritize retention efforts.

**Project Objective:** Build and deploy a classification model that identifies customers likely to leave, with particular attention to Recall for the churn class.

## Dataset

**Dataset:** Telco Customer Churn

The dataset contains information about customer demographics, subscribed services, contracts, and billing behavior.

### Main Feature Categories

- Demographics: Gender, SeniorCitizen, Partner, Dependents.
- Account Information: Tenure, Contract, PaymentMethod, PaperlessBilling.
- Services: PhoneService, InternetService, OnlineSecurity, TechSupport, StreamingTV, and others.
- Financial Information: MonthlyCharges and TotalCharges.

### Target Variable

`Churn` is the binary classification target:

- `0` — Customer stays.
- `1` — Customer churns.

The customerID column is excluded from model training.

## Project Workflow

```text
Telco Customer Churn Dataset
            |
            v
Exploratory Data Analysis
            |
            v
Data Cleaning and Preprocessing
            |
            v
Stratified Train/Test Split
            |
            v
Train Seven Classification Models
            |
            v
Compare Evaluation Metrics
            |
            v
Select Logistic Regression
            |
            v
GridSearchCV Hyperparameter Tuning
            |
            v
Final Model Evaluation
            |
            v
Save Pipeline Using Joblib
            |
            v
Streamlit Web Application
            |
            v
Customer Churn Prediction
```

## Data Preprocessing

The following preprocessing steps were performed:

1. Inspected missing values, duplicates, and data types.
2. Identified hidden whitespace in the TotalCharges column.
3. Converted TotalCharges into a numerical data type.
4. Replaced missing TotalCharges values with zero for customers with zero tenure.
5. Removed customerID from the feature matrix.
6. Encoded Churn as a binary target.
7. Split the dataset into 80% training and 20% testing sets using stratification.
8. Applied OneHotEncoder to categorical features.
9. Applied StandardScaler to numerical features for scale-sensitive algorithms.

### Preprocessing Pipelines

Two preprocessing configurations were used.

**Scaled Preprocessor**
- OneHotEncoder for categorical features.
- StandardScaler for numerical features.
- Applied to Logistic Regression, SVM, and KNN.

**Unscaled Preprocessor**
- OneHotEncoder for categorical features.
- Numerical features passed through unchanged.
- Applied to Decision Tree, Random Forest, AdaBoost, and XGBoost.

Scikit-learn Pipelines and ColumnTransformers were used to keep preprocessing and model training consistent and help prevent data leakage.

## Model Training and Comparison

Seven classification algorithms were trained and evaluated:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. K-Nearest Neighbors (KNN)
4. Decision Tree
5. Random Forest
6. AdaBoost
7. XGBoost

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Recall was prioritized because identifying actual churners is particularly important for customer retention.

### Baseline Model Performance

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8041 | 0.6522 | 0.5615 | 0.6034 |
| AdaBoost | 0.8027 | 0.6589 | 0.5321 | 0.5888 |
| KNN | 0.7615 | 0.5528 | 0.5321 | 0.5422 |
| SVM | 0.7991 | 0.6631 | 0.4947 | 0.5666 |
| XGBoost | 0.7821 | 0.6128 | 0.4866 | 0.5425 |
| Decision Tree | 0.7246 | 0.4814 | 0.4840 | 0.4827 |
| Random Forest | 0.7679 | 0.5842 | 0.4358 | 0.4992 |

Logistic Regression achieved the strongest baseline performance across Accuracy, Recall, and F1-score and was selected for hyperparameter tuning.

## Hyperparameter Tuning

GridSearchCV was used to optimize the Logistic Regression pipeline.

### Search Configuration

- Cross-validation: 5-fold StratifiedKFold
- Scoring metric: Recall
- Parameters: C, solver, and class_weight

### Selected Hyperparameters

| Parameter | Value |
|---|---|
| C | 10 |
| solver | lbfgs |
| class_weight | balanced |
| max_iter | 1000 |
| Mean CV Recall | 0.8027 |

Using balanced class weights helped the model identify more customers who actually churn.

## Final Model Evaluation

The optimized model was evaluated on the held-out test dataset.

### Final Test Performance

| Metric | Score |
|---|---:|
| Accuracy | 75.59% |
| Precision | 52.57% |
| Recall | 82.09% |
| F1-score | 64.09% |

### Baseline vs Optimized Model

| Metric | Baseline | Optimized |
|---|---:|---:|
| Accuracy | 80.41% | 75.59% |
| Precision | 65.22% | 52.57% |
| Recall | 56.15% | 82.09% |
| F1-score | 60.34% | 64.09% |

### Model Selection Rationale

The optimized model increased Recall significantly, allowing it to identify a larger proportion of customers who actually churn.

This improvement came with a reduction in Precision and Accuracy, indicating more false-positive churn predictions.

The optimized model was selected because identifying at-risk customers was prioritized over maximizing overall Accuracy.

This trade-off should be considered when using predictions for customer retention campaigns.

## Model Serialization

The final optimized pipeline was saved using Joblib.

**Model Path:**

`notebooks/model/best_churn_model.joblib`

The saved pipeline contains:

- Fitted OneHotEncoder
- Fitted StandardScaler
- Optimized Logistic Regression classifier

Saving the complete pipeline allows the Streamlit application to make predictions without repeating training or manually applying preprocessing.

## Streamlit Deployment

An interactive Streamlit web application was developed to allow users to predict customer churn.

### Application Features

- Organized customer information form.
- Customer demographics and account details.
- Billing and contract information.
- Phone and internet service inputs.
- Automatic handling of service-dependent feature values.
- Prediction using the saved Machine Learning pipeline.
- Clear churn or no-churn prediction result.

### Prediction Process

1. The user enters customer information.
2. The application creates a one-row Pandas DataFrame.
3. The saved pipeline automatically preprocesses the input.
4. Logistic Regression generates a prediction.
5. Streamlit displays the prediction result.

## Project Structure

```text
project_base/
|
|-- .gitignore
|-- README.md
|-- streamlit_app.py
|
|-- dataset/
|   |-- Telco-Customer-Churn.csv
|
|-- notebooks/
    |-- exp_notebook.ipynb
    |
    |-- model/
        |-- best_churn_model.joblib
```

## Technologies Used

**Programming Language**
- Python

**Data Analysis**
- Pandas
- NumPy

**Machine Learning**
- Scikit-learn
- XGBoost

**Visualization**
- Matplotlib

**Model Serialization**
- Joblib

**Web Application**
- Streamlit

**Development Tools**
- Jupyter Notebook
- VS Code
- Git
- GitHub

## Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Classification_Mini-Project_Telco_Customer_Churn_Prediction_and_Deployment.git
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Navigate to the Project Directory

```bash
cd Classification_Mini-Project_Telco_Customer_Churn_Prediction_and_Deployment
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost streamlit joblib matplotlib jupyter
```

### 5. Run the Streamlit Application

```bash
streamlit run streamlit_app.py
```

Open the local URL displayed in the terminal, enter customer information, and click **Predict Customer Churn**.

**Note:** For reliable model loading, use a compatible version of Scikit-learn with the version used to save the Joblib artifact.

## Project Deliverables

- Jupyter Notebook with preprocessing, training, model comparison, hyperparameter tuning, and evaluation.
- Saved optimized Machine Learning pipeline.
- Interactive Streamlit application.
- One-minute application demonstration video.

## Limitations and Future Improvements

- The model was trained on a single Telco Customer Churn dataset.
- High Recall comes at the cost of additional false-positive predictions.
- Performance on new customer populations has not been established.
- Probability threshold optimization and business cost-based evaluation could be explored in future work.

## Conclusion

This project demonstrates an end-to-end Machine Learning classification workflow, from data preprocessing and model comparison to optimization and interactive deployment.

Seven classification algorithms were evaluated, and Logistic Regression was selected for further optimization.

The final model achieved **82.09% Recall** and **64.09% F1-score** on the test dataset.

The optimized model was saved using Joblib and integrated into a Streamlit application, demonstrating how an experimental Machine Learning workflow can be converted into an interactive prediction application.
