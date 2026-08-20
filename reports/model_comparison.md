# Model Comparison and Selection

This document records the full model-selection process — from initial screening across a broad set of algorithms, to narrowing down candidates, hyperparameter tuning, and final evaluation on the held-out test set.

---

## 1. Initial Screening (5-Fold Cross-Validation)

Eleven regression models were evaluated using 5-fold cross-validation on the training set, with RMSE (on the log-transformed target) as the comparison metric.

| Model              | CV RMSE  |
|---------------------|----------|
| GradientBoosting     | 0.126819 |
| XGBoost              | 0.127791 |
| SVR                  | 0.139141 |
| ElasticNet           | 0.139424 |
| Lasso                | 0.139773 |
| Ridge                | 0.141597 |
| RandomForest         | 0.143181 |
| AdaBoost             | 0.167378 |
| LinearRegression     | 0.168291 |
| KNN                  | 0.169352 |
| DecisionTree         | 0.199429 |

**Observations:**
- Ensemble/boosting methods (**GradientBoosting**, **XGBoost**) clearly outperformed all other models, with the lowest CV RMSE by a noticeable margin.
- **SVR** and the linear/regularized models (**ElasticNet**, **Lasso**, **Ridge**) formed a closely clustered mid-tier, all within a narrow RMSE band (~0.139–0.142).
- **RandomForest** performed reasonably but did not match the boosting models.
- **AdaBoost**, **LinearRegression**, **KNN**, and especially **DecisionTree** performed noticeably worse, indicating they were either too simple (LinearRegression), too sensitive to variance (DecisionTree), or not well-suited to this dataset without further tuning.

**Decision:** Based on these results, the model set was narrowed down to the three strongest performers — **GradientBoosting**, **XGBoost**, and **SVR** — for further tuning. SVR was retained despite not being in the immediate top two because it represented a fundamentally different modeling approach (margin-based vs. tree-based) and was competitive with the linear models, leaving open the possibility that tuning could unlock stronger performance.

---

## 2. Hyperparameter Tuning

The three shortlisted models were tuned using cross-validated hyperparameter search. The table below compares each model's best cross-validation RMSE after tuning.

| Model              | Tuned CV RMSE |
|---------------------|---------------|
| SVR                  | 0.120481      |
| XGBoost              | 0.120807      |
| GradientBoosting     | 0.124213      |

**Observations:**
- All three models improved substantially after tuning, confirming that their default hyperparameters were under-optimized for this dataset.
- Notably, **the model ranking changed after tuning**: SVR, which was the weakest of the three models before tuning, became the best-performing model after tuning — narrowly ahead of XGBoost, and ahead of GradientBoosting, which had originally led the untuned comparison.
- This shift highlights that untuned performance is not a reliable indicator of a model's true potential, and that a fair comparison between models requires tuning each of them properly before drawing conclusions.

---

## 3. Final Evaluation on the Test Set

After tuning, each model was evaluated once on the held-out test set to obtain an unbiased estimate of real-world performance.

| Model              | Test RMSE   | Test MAE    | Test R²  |
|---------------------|-------------|-------------|----------|
| SVR                  | ₹25,747.99  | ₹14,338.36  | 0.9091   |
| XGBoost              | ₹25,866.97  | ₹15,230.91  | 0.9082   |
| GradientBoosting     | ₹28,424.81  | ₹16,063.41  | 0.9005   |

**Observations:**
- **SVR** achieved the best results across all three metrics: the lowest Test RMSE, the lowest Test MAE, and the highest Test R².
- **XGBoost** performed very closely to SVR on RMSE and R², but had a noticeably higher MAE, suggesting slightly larger average errors despite a similar overall error magnitude.
- **GradientBoosting**, while still strong (R² of 0.9005), trailed both SVR and XGBoost on every metric.
- The relative ordering of models on the test set matched the ordering seen in the tuned cross-validation results, reinforcing that the cross-validation process gave a reliable estimate of generalization performance.

---

## 4. Final Model Selection

**Selected Model: Support Vector Regression (SVR)**

SVR was selected as the final model because it achieved:
- The lowest cross-validation RMSE among all tuned models, and
- The best performance on the held-out test set across RMSE, MAE, and R².

This consistency between cross-validation and test-set performance gives confidence that SVR's advantage is genuine and not an artifact of a particular data split, making it the most reliable choice for deployment.

---

## 5. Summary of the Selection Process

1. **Broad screening:** 11 models were evaluated with default hyperparameters using 5-fold cross-validation.
2. **Narrowing down:** The top 3 performers by CV RMSE — GradientBoosting, XGBoost, and SVR — were shortlisted for tuning.
3. **Tuning:** Each shortlisted model was hyperparameter-tuned, which changed the ranking, with SVR moving from third to first place.
4. **Final testing:** All three tuned models were evaluated once on the untouched test set to avoid test-set leakage during model selection.
5. **Selection:** SVR was chosen as the final model based on its superior and consistent performance across both cross-validation and test evaluation.
