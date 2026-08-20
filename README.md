# House Price Prediction

A machine learning project that predicts residential house prices using property characteristics from the Ames Housing dataset. The notebook explores the data, prepares numerical and categorical features, compares regression algorithms, tunes a Support Vector Regression model, and saves the trained model for later use.

## Project Overview

The goal of this project is to estimate a house's sale price from features such as:

- Overall material and finish quality.
- Living area and basement area.
- Number of bedrooms, bathrooms, and rooms.
- Garage capacity and condition.
- Construction and remodeling years.
- Neighborhood, house style, foundation, and other categorical attributes.

`SalePrice` is the target variable.

## Dataset

The notebook uses the Ames Housing dataset, containing 1,460 observations and 81 columns, including the target column `SalePrice`.

Important dataset characteristics identified in the analysis include:

- Target: `SalePrice`.
- Numerical and categorical predictors.
- Missing values in several columns.
- Highly incomplete features such as `PoolQC`, `MiscFeature`, `Alley`, and `Fence`.
- Sale prices ranging from 34,900 to 755,000 in the loaded training data.

The notebook expects the dataset file to be available at the path used in the data-loading cell. Update that path if your local file has a different name or location.

## Workflow

The project follows this general workflow:

1. Import Python and machine learning libraries.
2. Load the housing data with pandas.
3. Inspect the dataset using previews, descriptive statistics, data types, and missing-value analysis.
4. Separate the predictors from `SalePrice`.
5. Split the data into training and testing sets.
6. Engineer additional house-related features.
7. Impute missing numerical and categorical values.
8. Scale numerical features.
9. One-hot encode categorical features.
10. Select useful features with a Lasso-based selector.
11. Train and compare several regression models.
12. Tune the selected model with randomized cross-validation.
13. Evaluate predictions using regression metrics.
14. Save the final trained pipeline with `pickle`.

## Feature Engineering

The notebook defines a custom `FeatureEngineer` transformer that creates additional variables, including:

- `TotalSF`: combined finished living and basement space.
- `TotalBathrooms`: combined full and half bathrooms using weighted contributions.
- `TotalPorchSF`: combined porch and outdoor living areas.
- `HouseAge`: age of the house at the time of sale.
- `RemodAge`: years since the most recent remodeling.
- `GarageAge`: garage age relative to the sale year.
- `HasGarage`: whether the property has a garage.
- `HasBsmt`: whether the property has a basement.
- `HasFireplace`: whether the property has a fireplace.
- `HasPool`: whether the property has a pool.

These features are intended to provide the models with more meaningful representations of property size, age, and amenities.

## Data Preprocessing

The preprocessing pipeline treats numerical and categorical columns separately.

### Numerical features

- Missing values are filled using the median.
- Features are standardized with `StandardScaler`.

### Categorical features

- Missing values are filled with the most frequent category or the constant value `None`, depending on the feature group.
- Categories are converted into numerical indicators using `OneHotEncoder`.
- Unknown categories are ignored during transformation so that the pipeline can handle new data more safely.

The transformations are implemented with scikit-learn `Pipeline` and `ColumnTransformer` objects to help prevent inconsistent preprocessing between training and prediction.

## Models Used

The notebook imports and evaluates multiple regression algorithms, including:

- Linear Regression.
- Ridge Regression.
- Lasso Regression.
- Elastic Net.
- Decision Tree Regressor.
- K-Nearest Neighbors Regressor.
- Random Forest Regressor.
- Gradient Boosting Regressor.
- AdaBoost Regressor.
- Support Vector Regression.
- XGBoost Regressor.

The final tuning section uses an RBF-kernel `SVR` model inside the complete preprocessing and feature-selection pipeline.

## Hyperparameter Tuning

`RandomizedSearchCV` is used to search for effective SVR hyperparameters.

The search includes:

- `C`: `[0.1, 1, 10, 50, 100]`.
- `epsilon`: `[0.001, 0.01, 0.05, 0.1, 0.2]`.
- `gamma`: `['scale', 'auto']`.
- `kernel`: `['rbf']`.

The search uses:

- 30 randomly selected parameter combinations.
- Five-fold shuffled cross-validation.
- `random_state=42` for reproducibility.
- Negative root mean squared error as the optimization score.
- `n_jobs=-1` to use available processor cores.

## Evaluation Metrics

The notebook uses the following metrics to evaluate model performance:

- **Mean Absolute Error (MAE):** average absolute difference between actual and predicted prices.
- **Mean Squared Error (MSE):** average squared prediction error.
- **Root Mean Squared Error (RMSE):** square root of MSE, expressed in the target variable's units.
- **R² score:** proportion of target variance explained by the model.

For house-price prediction, lower MAE and RMSE are better, while a higher R² score is better.

## Installation

Create a virtual environment if desired, then install the required dependencies:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Activate the environment on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

## Running the Notebook

1. Clone or download this project.
2. Place the dataset in the location expected by the notebook.
3. Install the dependencies listed above.
4. Start Jupyter:

```bash
jupyter notebook
```

5. Open `house_price_prediction.ipynb`.
6. Run the cells from top to bottom.

Running the cells in order is recommended because later sections depend on objects created during data loading, preprocessing, training, and tuning.

## Model Persistence

The notebook imports `pickle` and `os` to persist the trained model. Saving the complete pipeline is preferable to saving only the estimator because the pipeline also contains feature engineering, imputers, encoders, scaling, and feature selection.

A saved pipeline can be loaded later with:

```python
import pickle

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

predictions = model.predict(new_data)
```

The new input data must contain the predictor columns expected by the pipeline and must not include `SalePrice` when generating predictions.

## Example Prediction Pattern

```python
import pandas as pd
import pickle

new_house = pd.DataFrame([{
    "OverallQual": 7,
    "GrLivArea": 1800,
    "GarageCars": 2,
    "TotalBsmtSF": 1000,
    "YearBuilt": 2005,
    "YearRemodAdd": 2005,
    "YrSold": 2010
}])

with open("model.pkl", "rb") as file:
    trained_pipeline = pickle.load(file)

predicted_price = trained_pipeline.predict(new_house)
print(predicted_price)
```

The example is illustrative. In practice, provide all feature columns required by the fitted pipeline, or adapt the inference code to use the exact schema used during training.

## Project Structure

```text
.
├── house_price_prediction.ipynb
├── data/
│   └── train.csv
├── model.pkl
└── README.md
```

The exact dataset and model filenames may differ depending on the paths used in the notebook.

## Reproducibility

The notebook uses fixed random states in the train/test split, cross-validation, feature-selection estimator, and randomized search where applicable. Results can still vary slightly across library versions or hardware configurations.

## Limitations

- The model is trained on historical Ames Housing data and may not generalize to other cities or current housing markets.
- The notebook does not represent a production deployment or live property-valuation service.
- Predictions depend strongly on the quality and completeness of input features.
- A single train/test split may not fully represent out-of-sample performance.
- Model performance should be interpreted using multiple metrics rather than R² alone.

## Future Improvements

- Add a dedicated `requirements.txt` file with pinned package versions.
- Use log transformation of the `SalePrice` target to reduce skewness.
- Compare models using repeated cross-validation.
- Add residual analysis and prediction-error visualizations.
- Investigate outliers and influential observations.
- Build an inference script or Streamlit application.
- Add automated tests for the custom feature-engineering transformer.
- Track experiment results and tuned parameters in a structured results table.

## License

No license is specified in the notebook. Add an appropriate license before publicly distributing the project.