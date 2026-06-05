### White classifier applied to black individuals

| metric | white | black | delta |
| --- | --- | --- | --- |
| balanced accuracy | 0.90 | 0.67 | -0.23 |
| macro F1 | 0.89 | 0.66 | -0.23 |
| ROC AUC | 0.985 | 0.853 | -0.13 |

Performance drops a lot when the white model is applied to the black cohort. The drop is seen across all subtypes, but class 4 (Differentiated) drops the most: only 0.42 recall and 0.17 precision, meaning the model both misses true class-4 samples and over-assigns the label to others.

This is a population/domain-shift effect: differences in subtype expression patterns between cohorts break the transferred decision boundaries. To predict subtypes well across cohorts, the classifier should be trained on all data (Q9).