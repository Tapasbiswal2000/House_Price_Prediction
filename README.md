# 🏠 House Price Prediction

An end-to-end Machine Learning regression project that predicts residential house prices using property characteristics from the Kaggle House Prices dataset.

The project covers the complete machine learning workflow:

**EDA → Data Cleaning → Feature Engineering → Preprocessing → Model Selection → Cross-Validation → Hyperparameter Tuning → Evaluation → Model Saving**

---

## 📌 Project Overview

House price prediction is a regression problem where the objective is to predict the selling price of a house based on its characteristics.

The dataset contains information about:

* Overall quality
* Living area
* Basement area
* Garage
* Number of rooms
* Neighborhood
* Year built
* Property condition
* Exterior features
* And many other property attributes

The project compares multiple machine learning regression algorithms and selects the best-performing model using cross-validation and test-set evaluation.

The final selected model is **Support Vector Regression (SVR) with an RBF kernel**.

---

## 🎯 Objectives

The main objectives of this project are:

* Perform Exploratory Data Analysis (EDA)
* Understand numerical and categorical features
* Analyze missing values
* Detect potential outliers
* Analyze feature distributions and skewness
* Study relationships between features and house prices
* Perform feature engineering
* Handle numerical and categorical features separately
* Apply appropriate preprocessing
* Transform the skewed target variable
* Compare multiple regression algorithms
* Use 5-fold cross-validation for reliable model selection
* Tune the best-performing models
* Evaluate models on unseen test data
* Select the final model
* Save the complete preprocessing and prediction pipeline
* Prepare the model for future API deployment

---

# 📊 Dataset

The project uses the **Kaggle House Prices: Advanced Regression Techniques** dataset.

### Dataset Information

| Property         |       Value |
| ---------------- | ----------: |
| Rows             |       1,460 |
| Original Columns |          81 |
| Target Variable  | `SalePrice` |
| Input Features   |          80 |

The dataset contains both numerical and categorical features.

The target variable is:

```text
SalePrice
```

which represents the final selling price of each house.

---

# 🔎 Exploratory Data Analysis

The EDA phase was performed before model development.

The following areas were investigated:

* Dataset structure
* Data types
* Missing values
* Duplicate records
* Numerical feature distributions
* Categorical feature distributions
* Outliers
* Feature skewness
* Target distribution
* Feature correlations
* Numerical features vs `SalePrice`
* Categorical features vs `SalePrice`
* Important feature relationships
* Pair plots

---

## 🔍 Important Correlations

Some of the strongest correlations with `SalePrice` were:

| Feature     | Correlation with SalePrice |
| ----------- | -------------------------: |
| OverallQual |                      0.791 |
| GrLivArea   |                      0.709 |
| GarageCars  |                      0.640 |
| GarageArea  |                      0.623 |
| TotalBsmtSF |                      0.614 |

`OverallQual` had the strongest correlation with `SalePrice`.

This indicates that the overall quality of the house is one of the most important predictors of its selling price.

---

## 📈 Target Distribution

The original `SalePrice` distribution was positively skewed.

### Original target skewness

```text
1.883
```

Because of this skewness, a logarithmic transformation was applied.

```python
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)
```

After transformation:

```text
Skewness = 0.121
```

The log transformation significantly reduced the skewness of the target.

For final interpretation, predictions were converted back to the original price scale using:

```python
pred_dollars = np.expm1(pred_log)
```

---

# 🛠️ Feature Engineering

Feature engineering was performed after train-test splitting to maintain a leakage-aware workflow.

The original dataset contained 80 input features after removing the target.

Several meaningful features were created.

### Engineered Features

| Feature          | Description                       |
| ---------------- | --------------------------------- |
| `TotalSF`        | Combined total area               |
| `TotalBathrooms` | Combined bathroom representation  |
| `TotalPorchSF`   | Combined porch area               |
| `HouseAge`       | Age of the house                  |
| `RemodAge`       | Age since remodeling              |
| `HasGarage`      | Whether the house has a garage    |
| `HasBsmt`        | Whether the house has a basement  |
| `HasFireplace`   | Whether the house has a fireplace |
| `HasPool`        | Whether the house has a pool      |

Feature count increased from:

```text
80 → 89
```

After feature engineering:

```text
Numerical Features: 46
Categorical Features: 43
```

After preprocessing and encoding:

```text
X_train_processed: (1168, 310)
X_test_processed : (292, 310)
```

---

# 🧹 Data Preprocessing

Numerical and categorical features were processed separately.

## Numerical Features

Numerical features were processed and scaled where required by the machine learning models.

## Categorical Features

Categorical features were encoded so that machine learning algorithms could use them.

Special handling was applied to categorical features where a missing value represents the **absence of a particular house feature**.

Examples include:

* Garage
* Basement
* Fireplace
* Pool
* Fence
* Alley

These were handled separately from ordinary categorical missing values.

---

# 🔄 Train-Test Split

The dataset was divided into training and testing sets before data-dependent preprocessing.

```text
X_train: (1168, 80)
X_test : (292, 80)

y_train: (1168,)
y_test : (292,)
```

This corresponds to an approximately:

```text
80% Training
20% Testing
```

split.

The test set was kept separate and was only used for final model evaluation.

---

# ⚙️ Preprocessing Pipeline

A Scikit-learn `Pipeline` and `ColumnTransformer` were used to organize preprocessing and model training.

The pipeline ensured that preprocessing steps were consistently applied during:

* Training
* Cross-validation
* Testing
* Future predictions

This approach also helped prevent data leakage during model evaluation.

---

# 🤖 Models Compared

The following regression algorithms were initially evaluated:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet
5. Decision Tree
6. K-Nearest Neighbors
7. Random Forest
8. Gradient Boosting
9. AdaBoost
10. Support Vector Regression
11. XGBoost

---

# 🔄 Cross-Validation

Before selecting the final model, **5-fold K-Fold Cross-Validation** was performed.

The training data was divided into five folds.

Each model was trained and validated five times using different validation folds.

The main metric used for model comparison was:

```text
RMSE
```

on the log-transformed target.

### Why Cross-Validation?

A single train-test split can sometimes give an unreliable estimate of model performance.

Cross-validation provides a more reliable estimate by evaluating the model across multiple train-validation splits.

---

# 📊 Initial Model Comparison

| Rank | Model            |  CV RMSE |    CV R² |
| ---: | ---------------- | -------: | -------: |
|    1 | GradientBoosting | 0.126819 | 0.894098 |
|    2 | XGBoost          | 0.127791 | 0.892650 |
|    3 | SVR              | 0.139141 | 0.871244 |
|    4 | ElasticNet       | 0.139424 | 0.868744 |
|    5 | Lasso            | 0.139773 | 0.867988 |
|    6 | Ridge            | 0.141597 | 0.865588 |
|    7 | RandomForest     | 0.143181 | 0.865288 |
|    8 | AdaBoost         | 0.167378 | 0.814841 |
|    9 | LinearRegression | 0.168291 | 0.809038 |
|   10 | KNN              | 0.169352 | 0.810720 |
|   11 | DecisionTree     | 0.199429 | 0.738388 |

Based on the initial cross-validation results, the following three models were selected for further tuning:

* GradientBoosting
* XGBoost
* SVR

---

# 🎛️ Hyperparameter Tuning

Hyperparameter tuning was performed on the three strongest candidate models.

---

## GradientBoosting

### Best Parameters

```text
n_estimators = 500
min_samples_split = 10
min_samples_leaf = 2
max_depth = 3
learning_rate = 0.05
```

### Best CV RMSE

```text
0.124213
```

---

## XGBoost

### Best Parameters

```text
subsample = 0.8
n_estimators = 700
max_depth = 3
learning_rate = 0.05
colsample_bytree = 0.7
```

### Best CV RMSE

```text
0.120807
```

---

## SVR

### Best Parameters

```text
kernel = rbf
C = 1
gamma = auto
epsilon = 0.01
```

### Best CV RMSE

```text
0.120481
```

---

# 🏆 Tuned Model Comparison

After hyperparameter tuning:

| Rank | Model            | Best CV RMSE |
| ---: | ---------------- | -----------: |
|    1 | **SVR**          | **0.120481** |
|    2 | XGBoost          |     0.120807 |
|    3 | GradientBoosting |     0.124213 |

An important observation was that the model ranking changed after tuning.

### Before tuning

```text
GradientBoosting
       ↓
XGBoost
       ↓
SVR
```

### After tuning

```text
SVR
 ↓
XGBoost
 ↓
GradientBoosting
```

This demonstrates why hyperparameter tuning is important.

---

# 📈 Final Test Evaluation

After tuning, the three selected models were evaluated on the held-out test set.

| Rank | Model            | Test RMSE (Log) |      Test R² |      Test RMSE |       Test MAE |
| ---: | ---------------- | --------------: | -----------: | -------------: | -------------: |
|    1 | **SVR**          |    **0.130269** | **0.909062** | **₹25,747.99** | **₹14,338.36** |
|    2 | XGBoost          |        0.130860 |     0.908235 |     ₹25,866.97 |     ₹15,230.91 |
|    3 | GradientBoosting |        0.136290 |     0.900461 |     ₹28,424.81 |     ₹16,063.41 |

---

# 🥇 Final Model

## Support Vector Regression — SVR

The final model selected for the project is:

```text
SVR
Kernel: RBF
C: 1
Gamma: auto
Epsilon: 0.01
```

### Final Performance

| Metric          |         Result |
| --------------- | -------------: |
| CV RMSE         |   **0.120481** |
| Test RMSE (Log) |   **0.130269** |
| Test R²         |   **0.909062** |
| Test RMSE       | **₹25,747.99** |
| Test MAE        | **₹14,338.36** |

The final model achieved an R² score of approximately:

```text
90.9%
```

on the held-out test set.

The average absolute prediction error was approximately:

```text
₹14,338
```

---

# 📊 Model Diagnostics

Several visualizations were generated to understand the final model's behavior.

## Actual vs Predicted

The actual vs predicted plot compares:

```text
Actual SalePrice
        vs
Predicted SalePrice
```

A well-performing model should have predictions close to the diagonal reference line.

---

## Residual Distribution

Residuals were calculated as:

```text
Residual = Actual Price - Predicted Price
```

The residual distribution was analyzed to understand the overall pattern of prediction errors.

---

## Residuals vs Predicted

This visualization was used to identify potential patterns or systematic errors in the model's predictions.

Model performance plots are stored in:

```text
images/model_performance_plots/
```

---

# 💾 Model Saving

The complete trained preprocessing and prediction pipeline was saved using Pickle.

```text
models/final_house_price_model.pkl
```

The model was successfully loaded and verified as:

```text
sklearn.pipeline.Pipeline
```

Saving the complete pipeline is important because it stores the preprocessing steps together with the trained model.

This allows new input data to go through the same preprocessing workflow before generating predictions.

---

# 📁 Project Structure

```text
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
│       ├── final_svr_model.png
│       ├── model.png
│       └── svr_residuals_vs_predicted.png
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
```

---

# 🧰 Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost

### Development

* Jupyter Notebook
* VS Code

### Model Persistence

* Pickle

### Version Control

* Git
* GitHub

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Tapasbiswal2000/House_Price_Prediction.git
```

## 2. Navigate to the Project

```bash
cd House_Price_Prediction
```

## 3. Create a Virtual Environment

```bash
python -m venv venv
```

## 4. Activate the Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Open the Jupyter Notebook:

```text
notebooks/house_price_prediction.ipynb
```

Run the notebook cells sequentially to reproduce:

1. Data loading
2. Data inspection
3. Exploratory Data Analysis
4. Train-test splitting
5. Feature engineering
6. Data preprocessing
7. Target transformation
8. Model selection
9. Cross-validation
10. Hyperparameter tuning
11. Test evaluation
12. Model saving

---

# 📚 Reports

Detailed project documentation is available inside the `reports/` directory.

## Challenges Report

```text
reports/ML_Project_Challenges_Report.md
```

This report documents the major challenges faced during:

* Data preprocessing
* Feature engineering
* Pipeline construction
* Model selection
* Cross-validation
* Hyperparameter tuning
* Model evaluation
* Model persistence

## Model Comparison Report

```text
reports/model_comparison.md
```

This report contains detailed information about:

* Initial model comparison
* Cross-validation results
* Test-set evaluation
* Hyperparameter tuning
* Tuned model comparison
* Final model selection

---

# 🔮 Future Improvements

The current project focuses on the complete machine learning workflow.

The next stage of the project will focus on deployment.

### Planned Improvements

* Build a **FastAPI REST API**
* Load the saved Pickle model into the API
* Create prediction endpoints
* Add request validation
* Create a frontend prediction interface
* Connect frontend and backend
* Dockerize the complete application
* Add automated testing
* Add GitHub Actions CI/CD
* Deploy the application to the cloud
* Add model monitoring

---

# 🚀 Future Deployment Architecture

The planned architecture is:

```text
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
SVR Model
  │
  ▼
Predicted House Price
```

The saved model:

```text
models/final_house_price_model.pkl
```

will be loaded by the FastAPI backend to generate predictions for new house data.

---

# ⚠️ Important Notes

The model was trained using the Kaggle House Prices dataset and the preprocessing steps developed in this project.

New prediction data must contain the required input features expected by the saved pipeline.

The target variable:

```text
SalePrice
```

is not required as an input when making predictions.

---

# 👨‍💻 Author

**Tapas Ranjan Biswal**

GitHub:

https://github.com/Tapasbiswal2000

---

# ⭐ Conclusion

This project demonstrates a complete end-to-end Machine Learning regression workflow for house price prediction.

The workflow included:

```text
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
```

A total of **11 regression models** were initially compared.

The three strongest models were then tuned:

* SVR
* XGBoost
* GradientBoosting

After tuning and final test evaluation, **SVR with an RBF kernel** was selected as the final model.

### Final Results

| Metric    |         Result |
| --------- | -------------: |
| Test R²   |     **0.9091** |
| Test MAE  | **₹14,338.36** |
| Test RMSE | **₹25,747.99** |
| CV RMSE   |   **0.120481** |

The trained pipeline has been saved as:

```text
models/final_house_price_model.pkl
```

The project is now ready for the next phase:

**FastAPI Model Deployment → Docker → Cloud Deployment**
