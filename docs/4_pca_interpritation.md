### Task 4 Interpretation of PCA dimentional reduction

PCA shows weak separation. The first three components together
explain only ~19% of the variance (PC1 7.8%, PC2 6.1%, PC3 5.4%), so these
plots capture a small fraction of the total structure in the data.

Colored by race, black and white samples are fully mixed on all PC
plots, with no visible separation. This indicates that after log-transform
and standardization, ancestory is not a dominant variation and
there is no strong batch effect between the two groups.

Colored by ClusterK4_kmeans, separation is also weak but a trend is visible
on PC1 vs PC2: cluster 2 tends toward the lower-left, cluster 1 toward the
right, cluster 3 toward the upper-left, while cluster 4 is spread through the
center. Clusters overlap heavily and have no sharp boundaries. The other PC
pairs show little structure.

It is important to understand that weak separation in PCA does not imply the subtypes are not
separable. PCA is linear, which is limitation. A different dimentional reduction approach 
(tSNE, UMAP, latent space) or supervised classifier can therefore 
still perform well even though the clusters are not visually distinct here.