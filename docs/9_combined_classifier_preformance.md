### Q9c — Combined model: why better than Q8, slightly below Q7

| metric | white CV| white→black transfer | combined holdout |
| --- | --- | --- | --- |
| balanced accuracy | 0.90 | 0.67 | 0.84 |
| macro F1 | 0.89 | 0.66 | 0.84 |
| macro ROC AUC | 0.985 | 0.853 | 0.956 |

The combined model (0.84) performs much better than the white→black transfer (0.67) and slightly below the white-only CV (0.90).

**Better than transfer:** there is no domain shift between train and holdout — both are drawn from the same mixed black+white distribution, and the model has seen both cohorts during training unlike in white->black transfer where boundaries learned only on white did not align with black expression.

**Slightly below white-on-white:** not because of less data — the combined set is larger — but because the task is harder. The model must generalize across two populations with different subtype expression patterns and proportions instead of fitting one homogeneous group. This is the expected behaviour: lower accuracy on any single population in exchange for a model that can generilize accross cohorts.

The clearest evidence is Differentiated (class 4): it shows poor performance under transfer (recall 0.42, precision 0.17) but recovers in the combined model (recall 0.65, precision 0.72). It remains the weakest subtype — consistent with its small size and low kmeans/NMF concordance seen in EDA — but no longer fails.