### Model choice

The dataset has significantly more genes than samples (~4000 features vs ~250 samples, p >> n). In this case tree-based models (XGBoost, CatBoost etc) tend to overfit.

It is always counter intuitive to choose simple ML classifier, but I would start with L2 logistic regression:

- The p >> n setting needs regularization; L2 shrinks coefficients and stays stable when features outnumber samples.
- Ideally I think we need to run hyper parameter search accross multiple models
I would prefer using Optuna for that. but that is time consuming

With `class_weight='balanced'` we handle imbalanced clusters (Proliferative n=~40), and all preprocessing (log-transform, scaling) is inside the pipeline so it is fit on the training fold only during CV to make sure there is no leakage.

### Evaluation (5-fold stratified CV)

| metric | value |
| --- | --- |
| balanced accuracy | 0.90 |
| macro F1 | 0.89 |
| macro ROC AUC | 0.985 |
| accuracy | 0.90 |

Performance is strong and even across all four subtypes (per-class F1 0.86-0.92), with no class collapsing — class balancing clearly probably whathelped the smaller Proliferative class. The high ROC AUC relative to F1 means the model ranks samples by class probability very well.
