# Project Report: Challenges Faced During the Machine Learning Workflow

## Introduction

This report documents the key challenges encountered while building an end-to-end machine learning pipeline — from data preprocessing and feature engineering to model selection, hyperparameter tuning, and deployment. Each challenge is presented along with the situation that caused it, how it was resolved, and the takeaway gained from the experience.

---

## 1. Deciding the Correct Order of the ML Workflow

**Challenge:** Deciding the correct order in which to execute the steps of the ML workflow.

**What Happened:** Early on, preprocessing, feature engineering, encoding, and scaling were applied without a clear sequence, which led to inconsistent and sometimes invalid results, since some steps depended on others being completed first.

**Solution:** A structured pipeline order was established: data cleaning → feature engineering → encoding/preprocessing → scaling → model training → evaluation → tuning.

**What I Learned:** The order of operations in an ML workflow is not arbitrary — each step builds on the output of the previous one, and getting the sequence wrong can silently corrupt the data or the model's understanding of it.

---

## 2. Understanding Why Feature Engineering Had to Happen Before Preprocessing

**Challenge:** Understanding why feature engineering needed to occur before general preprocessing steps.

**What Happened:** Initially, preprocessing (such as encoding and scaling) was attempted before new features were created, which meant newly engineered features couldn't benefit from or align with the preprocessing steps, and some engineered features became inconsistent with already-transformed columns.

**Solution:** Feature engineering was moved earlier in the pipeline so that new features were derived from raw, untransformed data, and preprocessing was applied afterward to the final feature set.

**What I Learned:** Feature engineering should work with raw, interpretable data. Once preprocessing transforms values (scaling, encoding), it becomes harder to meaningfully derive new features from them.

---

## 3. Handling Categorical Features with Meaningful "None" Values

**Challenge:** Handling categorical columns where `"None"` was not missing data but an actual meaningful category (e.g., "No Basement", "No Garage").

**What Happened:** These `"None"` values were initially being treated as missing data and dropped or imputed incorrectly, which caused loss of important information about the absence of a feature.

**Solution:** These columns were explicitly identified, and `"None"` was treated as a valid categorical label rather than a null value, using explicit fill values instead of automatic imputation.

**What I Learned:** Not all missing-looking values are actually missing — domain context matters, and blindly applying imputation techniques can destroy meaningful signal in the data.

---

## 4. Separating Absence-Based Categorical Columns from Other Categorical Columns

**Challenge:** Distinguishing "absence-based" categorical columns (where `"None"` means the feature doesn't exist) from regular categorical columns (where missing values are genuinely unknown).

**What Happened:** Treating all categorical columns the same way led to inconsistent handling — some columns needed `"None"` as a category, while others needed proper imputation (like mode-based filling).

**Solution:** Categorical columns were split into two groups and handled with separate logic: absence-based columns were filled with `"None"`, while genuinely missing columns were imputed appropriately.

**What I Learned:** A one-size-fits-all approach to categorical data doesn't work. Each column's context needs to be understood individually before deciding how to treat missing values.

---

## 5. Managing the Increase in Features After One-Hot Encoding

**Challenge:** Handling the large increase in the number of features after one-hot encoding categorical variables.

**What Happened:** One-hot encoding caused the feature space to expand significantly, increasing model complexity, training time, and the risk of overfitting or multicollinearity.

**Solution:** The dimensionality increase was managed by reviewing feature importance after training, and by ensuring the encoding was applied consistently across training and test sets using a shared pipeline/transformer.

**What I Learned:** Encoding strategy has real consequences on model size and performance — high-cardinality categorical variables need to be handled carefully, and dimensionality growth should be anticipated, not discovered after the fact.

---

## 6. Understanding Why Scaling Is Necessary for Models Such as SVR

**Challenge:** Understanding why feature scaling was required for algorithms like Support Vector Regression (SVR), when tree-based models didn't need it.

**What Happened:** Without scaling, distance-based and margin-based models like SVR performed poorly, since features with larger numeric ranges dominated the model's calculations.

**Solution:** Standardization (e.g., `StandardScaler`) was applied to numerical features specifically for scale-sensitive models, while scaling was skipped for tree-based models where it wasn't necessary.

**What I Learned:** Not all models treat features equally — algorithms that rely on distances or gradients (like SVR, KNN, linear models) require scaled input, while tree-based models are scale-invariant.

---

## 7. Handling the Skewed `SalePrice` Target

**Challenge:** Dealing with a heavily right-skewed target variable, `SalePrice`.

**What Happened:** The skewed distribution violated the assumptions of several regression models and led to models being overly influenced by extreme high-value outliers, resulting in poor predictions for typical cases.

**Solution:** The target variable was transformed to reduce skewness before training, and predictions were transformed back to the original scale during evaluation.

**What I Learned:** Target distribution matters as much as feature distribution — many regression models perform better and produce more balanced errors when the target is closer to a normal distribution.

---

## 8. Understanding Why `log1p()` Was Used

**Challenge:** Understanding the specific reasoning behind using `log1p()` instead of a standard `log()` transformation.

**What Happened:** Applying a plain logarithmic transformation raised concerns about handling zero or near-zero values, which would cause undefined results (`log(0)` is undefined).

**Solution:** `log1p()` (which computes `log(1 + x)`) was used instead, since it safely handles zero values while still reducing skewness, and `expm1()` was used to reverse the transformation afterward.

**What I Learned:** Small implementation details, like choosing `log1p()` over `log()`, can prevent numerical errors and edge-case failures, and it's important to understand *why* a specific function is preferred, not just that it's commonly used.

---

## 9. Choosing Cross-Validation Before Model Selection

**Challenge:** Deciding to use cross-validation before finalizing model selection, rather than relying on a single train-test split.

**What Happened:** A single train-test split gave results that varied depending on how the data happened to be divided, making it hard to trust which model was genuinely performing best.

**Solution:** K-fold cross-validation was used to evaluate multiple models, providing a more stable and reliable estimate of performance before choosing which model(s) to move forward with.

**What I Learned:** Cross-validation reduces the risk of choosing a model based on a "lucky" or "unlucky" data split, and gives a more trustworthy comparison across candidate models.

---

## 10. Avoiding Unnecessary Reliance on the Test Set During Tuning

**Challenge:** Avoiding the temptation to check performance on the test set repeatedly during hyperparameter tuning.

**What Happened:** Early experimentation involved evaluating changes directly against the test set, which risked indirectly "leaking" test set information into the model selection process.

**Solution:** Tuning and model comparison were restricted to the training data using cross-validation, and the test set was reserved strictly for final evaluation after all decisions were finalized.

**What I Learned:** The test set should represent unseen, real-world data. Using it too early or too often — even unintentionally — compromises the integrity of the final performance evaluation.

---

## 11. Understanding Why Hyperparameter Tuning Changed the Model Ranking

**Challenge:** Understanding why the ranking of models changed after hyperparameter tuning, compared to their ranking with default parameters.

**What Happened:** A model that performed best with default settings was no longer the top performer once other models were tuned, which was initially confusing.

**Solution:** It was recognized that default hyperparameters don't reflect a model's true potential — proper tuning was applied consistently across all candidate models before making a final comparison.

**What I Learned:** Comparing models fairly requires tuning each of them appropriately; a model's "out-of-the-box" performance is not necessarily indicative of its best possible performance.

---

## 12. Saving the Complete Pipeline Rather Than Only the Model

**Challenge:** Deciding to save the entire preprocessing + model pipeline instead of just the trained model.

**What Happened:** Saving only the trained model initially caused issues, since new/incoming data would not automatically go through the same preprocessing, encoding, and scaling steps used during training.

**Solution:** The complete pipeline (preprocessing steps + trained model) was saved as a single object, ensuring that any new input data is transformed identically to how the training data was processed.

**What I Learned:** A model is only as reliable as the preprocessing it depends on. For consistent, reproducible predictions, the entire pipeline — not just the final estimator — needs to be persisted.

---

## 13. Verifying That the Pickle File Can Actually Be Loaded

**Challenge:** Confirming that the saved pickle file could be successfully loaded and used for predictions after saving.

**What Happened:** After saving the pipeline, there was uncertainty about whether the file was correctly serialized and whether it would produce the same results when reloaded in a fresh environment.

**Solution:** The pickle file was explicitly reloaded in a separate test script/session, and predictions were run on sample data to confirm the loaded pipeline behaved identically to the original.

**What I Learned:** Saving a file isn't the end of the process — verifying that it loads correctly and produces consistent results is a necessary final check before considering a model "deployment-ready."

---

## Conclusion

Working through this project surfaced challenges at nearly every stage of the machine learning workflow — from data understanding and feature engineering, to model evaluation, tuning, and deployment readiness. Resolving each challenge reinforced a deeper understanding of *why* each step in a standard ML pipeline exists, not just *how* to implement it. The most valuable lesson overall was that a machine learning workflow is not a checklist of isolated steps, but an interconnected sequence where decisions made early on (such as encoding or feature engineering order) directly affect the validity and reliability of everything that follows.
