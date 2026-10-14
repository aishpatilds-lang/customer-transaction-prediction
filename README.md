# Customer Transaction Prediction

## Project Overview
This project aims to predict whether a banking customer will make a specific transaction in the future, regardless of the transaction amount. The analysis uses anonymized customer data and machine learning classification models to identify potential transacting customers.

## Problem Statement
Banks need to identify customers who are likely to make future transactions to support better customer targeting and decision-making. This project analyzes historical customer data and builds predictive models to classify customers based on their likelihood of transacting.

## Dataset Description
The dataset contains:
- **200,000 rows** of customer records
- **ID_code:** Unique customer identifier
- **target:** Binary target variable (0 = no transaction, 1 = transaction)
- **var_0 to var_199:** 200 anonymized numerical features

Note: The dataset contains anonymized features whose individual business meanings are not disclosed. The dataset file is not included in this repository.

## Technologies Used
- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- XGBoost

## Project Workflow
1. **Data Loading and Quality Checks** – Examined data types, missing values, duplicate records and identifiers.
2. **Exploratory Data Analysis (EDA)** – Analyzed target distribution, feature distributions, correlations and statistical differences between target classes.
3. **Data Preprocessing** – Used stratified sampling, scaling where appropriate and Scikit-learn pipelines to help prevent data leakage.
4. **Model Building** – Compared Logistic Regression, Random Forest, Gradient Boosting and XGBoost.
5. **Hyperparameter Tuning** – Applied RandomizedSearchCV to tune XGBoost.
6. **Model Evaluation** – Evaluated models using ROC-AUC, precision, recall, F1-score and a confusion matrix.
7. **Feature Importance Analysis** – Compared XGBoost feature importance with statistical feature rankings.

## Key Findings
- The target variable is imbalanced, with approximately 90% negative and 10% positive cases.
- Accuracy alone is insufficient for evaluating performance because of class imbalance.
- The anonymized features show generally weak pairwise correlations, so predictive information is distributed across multiple features.
- Different models provide different precision-recall trade-offs.

## Model Performance

The following results are reported in the notebook's model comparison section:

| Model | ROC-AUC | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.852 | 0.278 | 0.757 | 0.407 |
| Tuned XGBoost | 0.842 | 0.515 | 0.387 | 0.442 |
| Random Forest | 0.760 | 0.435 | 0.145 | 0.217 |
| Gradient Boosting | 0.760 | 0.778 | 0.002 | 0.003 |

**Model selection:** Tuned XGBoost achieved the highest F1-score and precision among the compared models, while Logistic Regression achieved the highest ROC-AUC and recall. The preferred model depends on the bank's priorities and the cost of false positives versus false negatives.

*Note: These are the results documented in the notebook. They are not a guarantee of performance on new or unseen data.*

## Challenges and Solutions
- **Anonymized features:** Used statistical analysis, correlation analysis and feature importance to investigate predictive patterns.
- **Class imbalance:** Used stratified sampling and class-weighting strategies where supported by the selected models.
- **Computational cost:** Used stratified subsamples for model comparison and hyperparameter tuning to make experimentation more practical.
- **Model selection:** Compared multiple evaluation metrics instead of relying only on accuracy.

## How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/customer-transaction-prediction.git
cd customer-transaction-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset
Obtain the dataset from an authorized source and place the training CSV file in the project directory.

Update the `DATA_PATH` variable in the notebook to match the actual CSV filename and location.

### 4. Run the notebook
Open the notebook using Jupyter Notebook or JupyterLab and execute the cells in order.

```bash
jupyter notebook
```

## Future Improvements
- Evaluate promising models using the complete training dataset and a suitable validation strategy.
- Explore additional class-imbalance handling techniques.
- Tune decision thresholds according to business requirements.
- Validate model performance on genuinely unseen data.
- Assess probability calibration and model stability.

## Conclusion
This project demonstrates an end-to-end machine learning workflow for customer transaction prediction, from exploratory data analysis and preprocessing to model comparison and evaluation. It highlights the importance of class imbalance, metric selection and business-driven model selection in banking applications.

## Author
**Aishwarya Patil**

GitHub: [aishpatilds-lang](https://github.com/aishpatilds-lang)