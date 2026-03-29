## Files Description

**'extract_features.py'**

This script is responsible for computing the three distance-based features for each gene pair: normalized Levenshtein distance, k-mer geodesic (Hellinger) distance, and normalized compression distance (NCD). It takes as input the raw gene pairs and their corresponding sequences, and outputs a structured dataset where each pair is represented by these three features along with its label (orthologous / non-orthologous).

**'train_model.py'**

This script trains and evaluates machine learning models for the pairwise ortholog prediction task. Using the distance-based features computed in the previous step—normalized Levenshtein distance, k-mer geodesic (Hellinger) distance, and normalized compression distance (NCD)—it trains two classifiers, Logistic Regression and Random Forest, to predict whether a pair of genes is orthologous. The models are evaluated on validation and test datasets, and their performance is reported using standard metrics such as accuracy and ROC-AUC. The trained models can then be saved for later use.
