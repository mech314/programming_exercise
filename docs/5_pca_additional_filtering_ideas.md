### Additional cleaning / validation steps

I already applied log-transform and standardization to the dataset, since
non-normalized data showed abnormally high variance at PC1 (~80%), I'm guessing
that this variance comes from the skewed expression values.

Beyond the log-transform and standardization, given the PCA
results I would also check:

- a few samples sit far from the main clusters; I would inspect
  per-sample expression distributions to flag technical outliers (low RNA
  quality, extreme values).
- I would check if there are genes that are heavily zero-inflated; I would remove
  very low-expression features that add noise without signal.
- unfiltered metadata contains batch-like fields (study, version,
  etc.). I would test whether PCA separates by any of these batch effects
- Class balance: check the distribution of ClusterK4_kmeans within each race,
  since rare subtypes are hard to see in PCA and require stratification.
- Nonlinear embeddings: since PC1-3 capture only ~19% of variance, I'm really qurious
  how UMAP or tSNE would separate the samples.