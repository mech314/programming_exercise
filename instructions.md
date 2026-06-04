# Coding Exercise

## Overview

In this coding exercise, you will have 24 hours to complete the exercises written out below. You will need to provide either plots and/or written answers to each question.

You can use any programming language to complete this assignment. You can use Claude or chatGPT-like systems for this exercise, but you will be expected to explain your reasoning for each step of your code.

If you have any questions, please reach out to me via email: natalie.davidson@cuanschutz.edu

## Grading

You do not need to get every answer correct in order to pass this assignment. What I am looking for is:

1. **Code clarity**
   a. Clear and concise comments
   b. Use of functions
   c. Appropriate variable/function names
   d. Effective use of software libraries for machine learning and data cleaning
2. **Legible and aesthetically pleasing plots**
3. **Ability to follow directions**
4. **Ability to explain reasoning and model choices**
5. **Data cleaning/wrangling abilities**

## Data Description

In this folder, you will find another folder titled "data." Within "data" there are the following files:

- **supp_table_1_GlobalMAD_genelist.csv**
  - Gene names that can be used as features in downstream tasks
- **supp_table_2_CommonGenes_genelist.csv**
  - Another set of gene names that can be used as features in downstream tasks
- **supp_table_3_main_black_metadata_table.tsv**
  - Metadata for Black individuals
    - `ID`: ID used to index the gene expression file "supp_table_6_black_expr.tsv"
    - `Ran_in_way_pipeline`: If it is "TRUE" then you will use this sample in the downstream tasks
    - `ClusterK*_kmeans`: output from running Kmeans clustering on the gene expression data for different number of clusters, ranging from 2-4.
    - `ClusterK*_NMF`: output from running NMF clustering on the gene expression data for different number of clusters, ranging from 2-4.
    - `ClusterK4_kmeans_TCGA_names`: TCGA-specific cluster names for K=4 using Kmeans.
    - `external_HGSCsubtype_estimate`: Another version of cluster names for K=4
    - Ignore the rest of the columns
- **supp_table_4_main_white_metadata_table.tsv**
  - Metadata for White individuals
    - Columns have the same meaning as in the file "supp_table_3_main_black_metadata_table.tsv"
- **supp_table_6_black_expr.tsv**
  - Gene expression table for Black individuals.
  - Rows are genes (features), and the first column is the gene name.
  - Columns are samples.
- **supp_table_7_white_expr.tsv**
  - Gene expression table for White individuals.
  - Rows are genes (features), and the first column is the gene name.
  - Columns are samples.

## Exercise

1. Filter the gene expression tables to only the genes listed in "supp_table_1_GlobalMAD_genelist.csv"
   a. How many genes and samples are there before and after filtering?

2. How many genes are shared between `supp_table_1_GlobalMAD_genelist.csv` and `supp_table_2_CommonGenes_genelist.csv`?

3. Make a PCA plot of the expression data of both Black and White individuals together
   a. Plot PC1 vs PC2, PC2 vs PC3, PC1 vs PC3
   b. Color the points by the value in the respective metadata column `ClusterK4_kmeans`
   c. Color the points by the individual's race
   d. What is the proportion of variability explained by PC1, PC2, and PC3

4. Write out your interpretation of the plots.

5. Are there any other data cleaning/validation steps that could be valuable given the PCA plots? You do not have to implement these steps. Only provide a written description of other things you want to check.

6. How do the proportion of values in the column `ClusterK4_kmeans` differ between Black and White individuals?

7. Using a supervised approach, build a model that predicts the value of `ClusterK4_kmeans` using the White individual's gene expression data. You can use any classifier you like, but provide justification for your choice.
   a. Key things to keep in mind
      i. You don't have to use all the genes, you can use the pre-selected features in supp_table_1 or supp_table_2. You can also use your own feature selection method if you like.
      ii. Make sure you use cross-validation and that it is stratified relative to whatever features you believe are important. (Think about your answer to question 6)
   b. Evaluate the performance of your trained classifier

8. Apply your classifier to the Black individuals. Do not retrain your model.
   a. Evaluate the performance of the classifier

9. Train your model using all of the data (both Black and White individuals), but hold out 20% for the final evaluation
   a. Use a stratified data split for cross-validation and final evaluation
   b. Evaluate the performance of your trained classifier on your held-out set.
   c. Explain why this classifier works better or worse than the previous one (from question 7 and 8)