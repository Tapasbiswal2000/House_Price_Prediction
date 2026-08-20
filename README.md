# 🏠 House Price Prediction

An end-to-end Machine Learning regression project that predicts residential house prices using property characteristics from the Kaggle House Prices dataset.

The project covers the complete machine learning workflow:

**EDA → Data Cleaning → Feature Engineering → Preprocessing → Feature Selection → Model Selection → Cross-Validation → Hyperparameter Tuning → Evaluation → Model Saving**

---

## 📌 Project Overview

House price prediction is a regression problem where the objective is to predict the selling price of a house based on its characteristics.

The dataset contains information about:

- Overall quality
- Living area
- Basement area
- Garage
- Number of rooms
- Neighborhood
- Year built
- Property condition
- Exterior features
- And many other property attributes

The project compares multiple machine learning regression algorithms and selects the best-performing model using cross-validation and test-set evaluation.

After feature selection and hyperparameter tuning, the final selected model is **XGBoost Regressor**.

---

## 🎯 Objectives

The main objectives of this project are:

- Perform Exploratory Data Analysis (EDA)
- Understand numerical and categorical features
- Analyze missing values
- Detect potential outliers
- Analyze feature distributions and skewness
- Study relationships between features and house prices
- Perform feature engineering
- Handle numerical and categorical features separately
- Apply appropriate preprocessing
- Perform feature selection
- Reduce unnecessary features
- Transform the skewed target variable
- Compare multiple regression algorithms
- Use 5-fold cross-validation for reliable model selection
- Tune the best-performing models
- Evaluate models on unseen test data
- Select the final model
- Save the complete preprocessing and prediction pipeline
- Prepare the model for future API deployment

---

# 📊 Dataset

The project uses the **Kaggle House Prices: Advanced Regression Techniques** dataset.

## Dataset Information

| Property | Value |
|---|---:|
| Rows | 1,460 |
| Original Columns | 81 |
| Input Features | 80 |
| Target Variable | `SalePrice` |

The dataset contains both numerical and categorical features.

The target variable is:

```text
SalePrice
🔎 Exploratory Data Analysis

The EDA phase was performed before model development.

The following areas were investigated:

Dataset structure
Data types
Missing values
Duplicate records
Numerical feature distributions
Categorical feature distributions
Outliers
Feature skewness
Target distribution
Feature correlations
Numerical features vs SalePrice
Categorical features vs SalePrice
Important feature relationships
Pair plots
🔍 Important Correlations

Some of the strongest correlations with SalePrice were:

Feature	Correlation with SalePrice
OverallQual	0.791
GrLivArea	0.709
GarageCars	0.640
GarageArea	0.623
TotalBsmtSF	0.614

OverallQual had the strongest correlation with SalePrice.

This indicates that the overall quality of the house is one of the most important predictors of its selling price.

📈 Target Distribution

The original SalePrice distribution was positively skewed.

Original target skewness
1.883

Because of this skewness, a logarithmic transformation was applied:

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

After transformation:

Skewness = 0.121

The log transformation significantly reduced the skewness of the target.

For final interpretation, predictions were converted back to the original price scale using:

pred_dollars = np.expm1(pred_log)
🛠️ Feature Engineering

Feature engineering was performed after train-test splitting to maintain a leakage-aware workflow.

The original dataset contained 80 input features after removing the target.

Several meaningful features were created.

Engineered Features
Feature	Description
TotalSF	Combined total area
TotalBathrooms	Combined bathroom representation
TotalPorchSF	Combined porch area
HouseAge	Age of the house
RemodAge	Age since remodeling
HasGarage	Whether the house has a garage
HasBsmt	Whether the house has a basement
HasFireplace	Whether the house has a fireplace
HasPool	Whether the house has a pool
GarageAge	Age of the garage

Feature count increased from:

80 → 89

After feature engineering:

Numerical Features: 46
Categorical Features: 43

After preprocessing and encoding:

X_train_processed: (1168, 324)
X_test_processed : (292, 324)
🧹 Data Preprocessing

Numerical and categorical features were processed separately.

Numerical Features

Numerical features were processed and scaled where required by the machine learning models.

Categorical Features

Categorical features were encoded so that machine learning algorithms could use them.

Special handling was applied to categorical features where a missing value represents the absence of a particular house feature.

Examples include:

Garage
Basement
Fireplace
Pool
Fence
Alley

These were handled separately from ordinary categorical missing values.

🔄 Train-Test Split

The dataset was divided into training and testing sets before data-dependent preprocessing.

X_train: (1168, 80)


X_test : (292, 80)


y_train: (1168,)


y_test : (292,)

This corresponds to an approximately:

80% Training
20% Testing

split.

The test set was kept separate and was only used for final model evaluation.

⚙️ Preprocessing Pipeline

A Scikit-learn Pipeline and ColumnTransformer were used to organize preprocessing and model training.

The pipeline ensured that preprocessing steps were consistently applied during:

Training
Cross-validation
Testing
Future predictions

This approach also helped prevent data leakage during model evaluation.

🎯 Feature Selection

Feature selection was introduced after preprocessing and feature engineering.

The project used LassoCV to identify important features.

LassoCV(
    alphas=np.logspace(-4, 0, 50),
    cv=5,
    max_iter=20000
)

The best regularization parameter identified by LassoCV was:

Alpha = 0.0005428675

The Lasso coefficients were used to identify the most important features.

Feature Reduction

Before feature selection:

324 features

After feature selection:

115 features

Therefore:

209 features were removed
Feature Reduction Summary
Stage	Number of Features
Original input features	80
After feature engineering	89
After preprocessing & encoding	324
After feature selection	115
Features removed	209

The feature selection process reduced the processed feature space by approximately 64.5%.

This produced a more compact feature set for model training.

🤖 Models Compared

The following regression algorithms were initially evaluated:

Linear Regression
Ridge Regression
Lasso Regression
ElasticNet
Decision Tree
K-Nearest Neighbors
Random Forest
Gradient Boosting
AdaBoost
Support Vector Regression
XGBoost
🔄 Cross-Validation

After feature selection, 5-fold K-Fold Cross-Validation was performed.

The training data was divided into five folds.

Each model was trained and validated five times using different validation folds.

The main metric used for model comparison was:

RMSE

on the log-transformed target.

Why Cross-Validation?

A single train-test split can sometimes give an unreliable estimate of model performance.

Cross-validation provides a more reliable estimate by evaluating the model across multiple train-validation splits.

📊 Model Comparison After Feature Selection
Rank	Model	CV RMSE	CV R²
1	Lasso	0.110808	0.919380
2	ElasticNet	0.111011	0.919122
3	Ridge	0.112356	0.917104
4	LinearRegression	0.116467	0.910950
5	GradientBoosting	0.119225	0.906787
6	XGBoost	0.120662	0.904481
7	RandomForest	0.133958	0.882311
8	SVR	0.147331	0.856608
9	AdaBoost	0.160639	0.830050
10	KNN	0.162352	0.826979
11	DecisionTree	0.192414	0.755303

Based on the initial post-feature-selection cross-validation results, Lasso, ElasticNet, Ridge, GradientBoosting and XGBoost performed strongly.

The strongest candidate models selected for further tuning were:

GradientBoosting
XGBoost
SVR
🎛️ Hyperparameter Tuning

Hyperparameter tuning was performed on the three strongest candidate models:

GradientBoosting
XGBoost
SVR

RandomizedSearchCV with 5-fold cross-validation was used.

GradientBoosting
Best Parameters
n_estimators = 500
min_samples_split = 10
min_samples_leaf = 2
max_depth = 3
learning_rate = 0.05
Best CV RMSE
0.119491
XGBoost
Best Parameters
subsample = 0.8
n_estimators = 700
max_depth = 3
learning_rate = 0.05
colsample_bytree = 0.7
Best CV RMSE
0.115379
SVR
Best Parameters
kernel = rbf
C = 1
gamma = auto
epsilon = 0.01
Best CV RMSE
0.123614
🏆 Tuned Model Comparison

After hyperparameter tuning:

Rank	Model	Best CV RMSE
1	XGBoost	0.115379
2	GradientBoosting	0.119491
3	SVR	0.123614

XGBoost achieved the lowest cross-validation RMSE among the tuned models.

📈 Final Test Evaluation

After tuning, the three selected models were evaluated on the held-out test set.

Rank	Model	Test RMSE (Log)	Test R²	Test RMSE	Test MAE
1	XGBoost	0.124330	0.917165	₹23,312.89	₹14,054.85
2	GradientBoosting	0.130257	0.909078	₹25,345.93	₹14,867.50
3	SVR	0.134451	0.903130	₹27,511.14	₹14,879.23
🥇 Final Model
XGBoost Regressor

The final model selected for the project is:

XGBoost Regressor
Best Parameters
n_estimators = 700


learning_rate = 0.05


max_depth = 3


subsample = 0.8


colsample_bytree = 0.7
Final Performance
Metric	Result
CV RMSE	0.115379
Test RMSE (Log)	0.124330
Test R²	0.917165
Test RMSE	₹23,312.89
Test MAE	₹14,054.85

The final model achieved an R² score of approximately:

91.7%

on the held-out test set.

The average absolute prediction error was approximately:

₹14,055
📊 Model Diagnostics

Several visualizations were generated to understand the final model's behavior.

Actual vs Predicted

The actual vs predicted plot compares:

Actual SalePrice
        vs
Predicted SalePrice

A well-performing model should have predictions close to the diagonal reference line.

Residual Distribution

Residuals were calculated as:

Residual = Actual Price - Predicted Price

The residual distribution was analyzed to understand the overall pattern of prediction errors.

Residuals vs Predicted

This visualization was used to identify potential patterns or systematic errors in the model's predictions.

Model performance plots are stored in:

images/model_performance_plots/
💾 Model Saving

The complete trained preprocessing and prediction pipeline is saved using Pickle.

models/final_house_price_model.pkl

The intended final pipeline contains:

Feature Engineering
        ↓
Preprocessing
        ↓
Feature Selection
        ↓
XGBoost Model

Saving the complete pipeline allows new input data to go through the same preprocessing and feature-selection workflow before generating predictions.

📁 Project Structure
House_Price_Prediction/


│
├── data/
│   └── raw/
│       └── data.csv
│
├── images/
│   ├── eda_plots/
│   │   ├── Boxplots_numerical_feature.png
│   │   ├── Corelation_top_feature.png
│   │   ├── Correlation_of_numericalFeature_with_SalePrice.png
│   │   ├── Distributions_of_Numerical_Features.png
│   │   ├── categorical_feature.png
│   │   ├── categorical_vs_saleprice.png
│   │   ├── key_features_pairplot.png
│   │   ├── numeric_vs_saleprice.png
│   │   ├── saleprice_boxplot.png
│   │   ├── saleprice_by_neighborhood.png
│   │   ├── saleprice_distribution.png
│   │   ├── saleprice_log_comparison.png
│   │   └── skewed_features.png
│   │
│   └── model_performance_plots/
│       ├── best_model.png
│       ├── model.png
│       └── ...
│
├── models/
│   └── final_house_price_model.pkl
│
├── notebooks/
│   └── house_price_prediction.ipynb
│
├── reports/
│   ├── ML_Project_Challenges_Report.md
│   └── model_comparison.md
│
├── requirements.txt
│
└── README.md
🧰 Technologies Used
Programming Language
Python
Data Analysis
Pandas
NumPy
Data Visualization
Matplotlib
Seaborn
Machine Learning
Scikit-learn
XGBoost
Development
Jupyter Notebook
VS Code
Model Persistence
Pickle
Version Control
Git
GitHub
📦 Installation
1. Clone the Repository
git clone https://github.com/Tapasbiswal2000/House_Price_Prediction.git
2. Navigate to the Project
cd House_Price_Prediction
3. Create a Virtual Environment
python -m venv venv
4. Activate the Virtual Environment
Windows
venv\Scripts\activate
5. Install Dependencies
pip install -r requirements.txt
▶️ Running the Project

Open the Jupyter Notebook:

notebooks/house_price_prediction.ipynb

Run the notebook cells sequentially to reproduce:

Data loading
Data inspection
Exploratory Data Analysis
Train-test splitting
Feature engineering
Data preprocessing
Target transformation
Feature selection
Model selection
Cross-validation
Hyperparameter tuning
Test evaluation
Final model selection
Model saving
📚 Reports

Detailed project documentation is available inside the reports/ directory.

Challenges Report
reports/ML_Project_Challenges_Report.md

This report documents the major challenges faced during:

Data preprocessing
Feature engineering
Feature selection
Pipeline construction
Model selection
Cross-validation
Hyperparameter tuning
Model evaluation
Model persistence
Model Comparison Report
reports/model_comparison.md

This report contains detailed information about:

Initial model comparison
Feature selection
Cross-validation results
Test-set evaluation
Hyperparameter tuning
Tuned model comparison
Final model selection
🔮 Future Improvements

The current project focuses on the complete machine learning workflow.

The next stage of the project will focus on deployment.

Planned Improvements
Build a FastAPI REST API
Load the saved Pickle model into the API
Create prediction endpoints
Add request validation
Create a frontend prediction interface
Connect frontend and backend
Dockerize the complete application
Add automated testing
Add GitHub Actions CI/CD
Deploy the application to the cloud
Add model monitoring
🚀 Future Deployment Architecture

The planned architecture is:

User
  │
  ▼
Frontend
  │
  ▼
FastAPI
  │
  ▼
Preprocessing Pipeline
  │
  ▼
Feature Selection
  │
  ▼
XGBoost Model
  │
  ▼
Predicted House Price

The saved model:

models/final_house_price_model.pkl

will be loaded by the FastAPI backend to generate predictions for new house data.

⚠️ Important Notes

The model was trained using the Kaggle House Prices dataset and the preprocessing, feature engineering, and feature-selection pipeline developed in this project.

New prediction data must contain the required input features expected by the saved pipeline.

The target variable:

SalePrice

is not required as an input when making predictions.

👨‍💻 Author

Tapas Ranjan Biswal

GitHub:

https://github.com/Tapasbiswal2000/House_Price_Prediction

⭐ Conclusion

This project demonstrates a complete end-to-end Machine Learning regression workflow for house price prediction.

The workflow included:

Data Collection
      ↓
Exploratory Data Analysis
      ↓
Train-Test Split
      ↓
Feature Engineering
      ↓
Data Preprocessing
      ↓
Target Transformation
      ↓
Feature Selection
      ↓
Model Selection
      ↓
5-Fold Cross-Validation
      ↓
Hyperparameter Tuning
      ↓
Final Test Evaluation
      ↓
Final Model Selection
      ↓
Model Saving

A total of 11 regression models were initially compared.

Feature selection using Lasso reduced the processed feature space from:

324 → 115 features

This removed 209 features, reducing the processed feature space by approximately 64.5%.

The three strongest models selected for hyperparameter tuning were:

GradientBoosting
XGBoost
SVR

After hyperparameter tuning and final test evaluation, XGBoost was selected as the final model.

Final Results
Metric	Result
Test R²	0.9172
Test MAE	₹14,054.85
Test RMSE	₹23,312.89
CV RMSE	0.115379

The trained preprocessing, feature-selection, and XGBoost pipeline is saved as:

models/final_house_price_model.pkl

The project is now ready for the next phase:

FastAPI Model Deployment → Docker → Cloud Deployment