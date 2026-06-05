### White classifier applied to black individuals

| metric            | white     | black           | Δ     |
|-------------------|-----------|------------------|-------|
| balanced accuracy | 0.90      | 0.67             | -0.23 |
| F1                | 0.89      | 0.66             | -0.23 |
| ROC AUC           | 0.985     | 0.853            | -0.13 |

Performance drops a lot when the white model is applied to the black
cohort. Drop in performance could be seen across all sybtypes, but
class 4 shows highes drop, with only 0.42 true. 

This is a population/domain-shift effect, differences in subtype
expression patterns break the transferred boundaries. To achieve better results
in predicting subtypes accross different cohort classifier should be trained on all data