# Adult Income — Day 5 Final Submission

## Objective
Build a reproducible binary classifier for whether an individual's annual income is `>50K` or `<=50K`. The primary metric is **F1** because the positive class is the minority class and both precision and recall matter.

## Data and split
- Source: Adult Income dataset (`adult.data`)
- Rows: 32,561
- Features: 14
- Target: `income`
- Missing marker `?` replaced with NaN.
- Stratified split with `random_state=42`: 60% train (19,536), 20% development (6,512), 20% final test (6,513).

## Leak-safe preprocessing
Numeric: median imputation + StandardScaler.
Categorical: most-frequent imputation + OneHotEncoder(handle_unknown='ignore').
All preprocessing is inside the sklearn Pipeline/ColumnTransformer.

## Models
Compared:
1. Logistic Regression
2. Random Forest
3. Gradient Boosting

Model selection used 5-fold StratifiedKFold (`random_state=42`) and F1 on the training partition. Hyperparameter search was performed for Logistic Regression and Random Forest; Gradient Boosting was evaluated as a fixed, documented candidate.

Mean 5-fold CV F1:
- Logistic Regression: 0.6811
- Random Forest: 0.6784
- Gradient Boosting: 0.6792

Selected model: **Logistic Regression**.

## Threshold
The decision threshold was selected **only on the development set**, maximizing F1. Selected threshold: **0.66**.
The final model was then refit on train + development data. The test set was kept untouched until the final evaluation.

## Final test results
- Accuracy: 0.8408
- Precision: 0.6595
- Recall: 0.7011
- F1: 0.6796
- ROC-AUC: 0.9016
- PR-AUC: 0.7597
- Brier score: 0.1308

Confusion matrix: TN=4376, FP=568, FN=469, TP=1100.

## Error analysis
False positives are cases predicted `>50K` but actually `<=50K`; false negatives are cases predicted `<=50K` but actually `>50K`. See the two CSV files in `results/`.

## Reproducibility
The final reusable artifact is `adult_income_final_pipeline.joblib`. It contains the fitted preprocessing + classifier, selected threshold, feature lists, target mapping, and reproducibility metadata. Loading the artifact and predicting on the same schema reproduces the saved decision rule.

## Files
- Executed notebook and PDF
- README and requirements
- Raw `adult.data`
- Reusable Joblib model
- Inference script
- Results CSVs
- Figures PNGs
- Final report PDF

## Important leakage rule
No final-test result was used for model selection, hyperparameter selection, or threshold selection. The final test set is evaluated once after all decisions are frozen.
