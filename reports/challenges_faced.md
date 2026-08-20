# Challenges Faced

This document describes the main challenges encountered while developing the house-price prediction project in `house_price_prediction.ipynb` and the approaches used to address them.

## 1. High-Dimensional Dataset

The dataset contains 1,460 rows and 81 columns. These columns include numerical measurements, ordinal-quality variables, nominal categories, and features describing the absence of facilities.

Working with many columns increases the risk of:

- Overfitting.
- Redundant information.
- Slow model training.
- Difficult interpretation.
- A very large encoded feature matrix.

After feature engineering and one-hot encoding, the project produced 324 transformed features. This made feature selection an important part of the workflow.

### Approach Used

A Lasso-based feature-selection process was used to identify useful predictors. The final selector reduced the feature space from 324 features to 115 features, dropping 209 features with zero or negligible importance.

## 2. Missing Values

Several columns contain missing values. Some missing values represent genuinely unavailable house features rather than random data-entry errors. For example, a missing value in a garage-quality column may mean that the property does not have a garage.

The most incomplete columns include:

- `PoolQC`.
- `MiscFeature`.
- `Alley`.
- `Fence`.
- `MasVnrType`.
- `FireplaceQu`.
- `LotFrontage`.

Simply removing all rows or columns containing missing values would cause unnecessary information loss.

### Approach Used

The preprocessing pipeline uses different strategies based on the feature type and meaning of the missing value:

- Numerical values are imputed with the median.
- Absence-related categorical columns are filled with `None`.
- Other categorical columns are imputed with their most frequent value.
- One-hot encoding uses `handle_unknown="ignore"` so new categories do not break prediction.

This approach preserves rows while distinguishing between an absent facility and an unknown category where appropriate.

## 3. Mixed Numerical and Categorical Features

The dataset combines continuous variables such as `LotArea` and `GrLivArea` with categorical variables such as `Neighborhood`, `HouseStyle`, `Foundation`, and `SaleCondition`.

These feature types cannot be processed identically. Numerical variables require imputation and scaling, while categorical variables must be converted into numerical representations.

### Approach Used

A `ColumnTransformer` was created with separate pipelines for:

- Numerical features: median imputation followed by standardization.
- Absence-related categorical features: constant imputation followed by one-hot encoding.
- Other categorical features: most-frequent imputation followed by one-hot encoding.

Keeping these operations inside a scikit-learn pipeline helps ensure that the same transformations are applied during training, validation, and prediction.

## 4. Feature Engineering Complexity

Raw columns do not always express the complete meaning of a property. For example, a house's overall usable area may depend on multiple floor and basement columns, while house age depends on both construction year and sale year.

The project also needed to avoid invalid values. Age-related features can become negative if dates are inconsistent, and garage-year values may be missing when a property has no garage.

### Approach Used

A custom `FeatureEngineer` transformer creates meaningful features such as:

- `TotalSF`.
- `TotalBathrooms`.
- `TotalPorchSF`.
- `HouseAge`.
- `RemodAge`.
- `GarageAge`.
- `HasGarage`.
- `HasBsmt`.
- `HasFireplace`.
- `HasPool`.

Age values are clipped at zero, and missing `GarageYrBlt` values are handled before calculating garage age. The custom transformer is included in the pipeline so feature creation remains consistent for unseen data.

## 5. Outliers

House-price datasets may contain unusual properties whose characteristics differ greatly from most homes. Extremely large living areas combined with comparatively low sale prices can have a disproportionate effect on regression models.

The notebook identified two known `GrLivArea`/`SalePrice` outliers in the training set using the condition:

```python
GrLivArea > 4000 and SalePrice < 300000
```

### Approach Used

The two identified outliers were removed from the training set only. The test set was not modified, which avoids using test-set information to influence training decisions.

After removal, the training data contained 1,166 rows, while the test data contained 292 rows.

## 6. Skewed Target Variable

`SalePrice` is not normally distributed and contains a long right tail caused by expensive properties. A highly skewed target can make model training more difficult and can cause high-priced observations to dominate the loss.

### Approach Used

The notebook applies a log transformation using:

```python
ytrainlog = np.log1p(ytrain)
ytestlog = np.log1p(ytest)
```

Using `log1p` is safe for non-negative target values and compresses the effect of very large prices. Predictions should be converted back to the original price scale with `np.expm1` when interpreting final house prices.

## 7. Preventing Data Leakage

Preprocessing the complete dataset before splitting it can cause information from the validation or test set to influence the training process. This can produce overly optimistic evaluation results.

### Approach Used

The notebook follows a train/test split and fits preprocessing and feature-selection components on the training data before transforming the test data. Cross-validation is performed within pipelines so transformations are fitted separately inside each training fold.

Outlier removal is also performed on the training set only.

## 8. Choosing an Appropriate Model

No single regression algorithm is guaranteed to work best for every dataset. The project evaluates linear, regularized, tree-based, boosting, nearest-neighbor, support-vector, and XGBoost models.

The cross-validation results show meaningful differences between models. Lasso performed best among the tested models in the initial comparison, with an average CV RMSE of approximately 0.1108 and an average CV R² of approximately 0.9194 on the log-transformed target.

### Approach Used

The project compares models using five-fold shuffled cross-validation with:

- Root mean squared error.
- R² score.
- Mean and standard deviation across folds.

This provides a more reliable comparison than relying on one train/test result.

## 9. Hyperparameter Tuning

Model performance depends on selecting suitable hyperparameters. For the RBF-kernel SVR model, important parameters include `C`, `epsilon`, and `gamma`. Searching every possible combination can be computationally expensive.

### Approach Used

`RandomizedSearchCV` searches 30 randomly selected combinations using five-fold cross-validation. The search uses negative RMSE as its scoring function and parallel processing with `n_jobs=-1`.

The search space includes:

- `C`: 0.1, 1, 10, 50, and 100.
- `epsilon`: 0.001, 0.01, 0.05, 0.1, and 0.2.
- `gamma`: `scale` and `auto`.
- `kernel`: `rbf`.

## 10. Model Interpretation

After one-hot encoding, the model no longer works with only the original human-readable columns. A single original categorical variable can become many binary features, making interpretation more difficult.

The Lasso selector provides a way to inspect influential transformed features. The notebook identifies features related to neighborhood, exterior quality, living area, total area, functionality, zoning, and overall quality among the top-ranked predictors.

### Approach Used

The project records transformed feature names and examines Lasso coefficients. This helps connect model behavior to property characteristics, although coefficients should be interpreted carefully because they are calculated after scaling and encoding.

## 11. Maintaining a Reusable Prediction Pipeline

Saving only the final estimator is not enough because future inputs must undergo the same feature engineering, imputation, encoding, scaling, and feature selection used during training.

### Approach Used

The project constructs a complete scikit-learn pipeline and imports `pickle` for model persistence. Saving the complete pipeline makes it possible to reuse the same preprocessing steps during inference.

The saved model must be loaded with compatible versions of Python and the libraries used to train it.

## Lessons Learned

The project highlights several important machine learning practices:

- Understand the meaning of missing values before imputing them.
- Separate training and test data before fitting transformations.
- Use pipelines to make preprocessing reproducible.
- Engineer features that represent domain knowledge.
- Treat outliers carefully instead of removing observations automatically.
- Transform skewed target variables when appropriate.
- Compare multiple models using cross-validation.
- Use feature selection when encoding creates a large feature space.
- Save the complete preprocessing and modeling pipeline for deployment.

## Future Improvements

Possible improvements include:

- Use repeated cross-validation for more stable performance estimates.
- Compare final models on the original price scale after reversing the log transformation.
- Add residual plots and error analysis by neighborhood and price range.
- Test robust approaches to outlier detection rather than relying on one manually defined rule.
- Tune the best-performing Lasso, Elastic Net, Gradient Boosting, and XGBoost models as well as SVR.
- Add automated tests for `FeatureEngineer`.
- Pin dependency versions in a `requirements.txt` file.
- Build a prediction interface using Streamlit or a lightweight API.
