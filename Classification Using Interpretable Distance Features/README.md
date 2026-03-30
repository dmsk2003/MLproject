## Files Description

**'build_pairwise_dataset.py'**

This script generates the pairwise dataset used for the ortholog prediction task. It takes as input a gene-to-OG mapping file and precomputed gene embeddings, and constructs labeled gene pairs indicating whether two genes belong to the same orthogroup or not. The data is split at the OG level into training, validation, and test sets to prevent data leakage.
For the training set, the script applies a hard mining strategy to generate more informative examples, including hard positives (low similarity within the same OG) and hard negatives (high similarity across different OG). For the validation and test sets, pairs are sampled randomly to reflect a more natural distribution.
The output consists of three CSV files:
- `train_pairs.csv`
- `val_pairs.csv`
- `test_pairs.csv`

Each row represents a gene pair with its corresponding OG labels and binary classification label.

> **Note:** If the pairwise dataset files (`train_pairs.csv`, `val_pairs.csv`, `test_pairs.csv`) are already available, you can skip running this script.

**'extract_features.py'**

This script is responsible for computing the three distance-based features for each gene pair: normalized Levenshtein distance, k-mer geodesic (Hellinger) distance, and normalized compression distance (NCD). It takes as input the raw gene pairs and their corresponding sequences, and outputs a structured dataset where each pair is represented by these three features along with its label (orthologous / non-orthologous).

**'train_model.py'**

This script trains and evaluates machine learning models for the pairwise ortholog prediction task. Using the distance-based features computed in the previous step—normalized Levenshtein distance, k-mer geodesic (Hellinger) distance, and normalized compression distance (NCD)—it trains two classifiers, Logistic Regression and Random Forest, to predict whether a pair of genes is orthologous. The models are evaluated on validation and test datasets, and their performance is reported using standard metrics such as accuracy and ROC-AUC. The trained models can then be saved for later use.
