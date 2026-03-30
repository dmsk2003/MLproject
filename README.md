# Machine Learning project

In this project, we address the problem of predicting gene orthology using machine learning techniques. Understanding relationships between genes across different species is a fundamental challenge in computational biology, as it provides insights into evolution, gene function, and biological processes.

## Project Overview

This project is divided into two main parts, addressing the problem of orthologous gene identification at different levels:

1. **Part I – Pairwise Ortholog Classification**  
   In this stage, the goal is to determine whether two genes are orthologous.  
   Using ESM-2 embeddings, we compute similarity-based features and train machine learning models (Logistic Regression and Random Forest) to classify gene pairs as orthologous or non-orthologous.

2. **Interpretable Distance-Based Classification**  
   In addition to embedding-based methods, we explore interpretable sequence similarity measures (e.g., Levenshtein, Hellinger, NCD) for pairwise ortholog prediction, demonstrating that simple, explainable features can achieve strong performance.

3. **Part II – Inferring Orthologous Groups from Gene Sets**  
   Given a set of genes with unknown structure, the goal is to:
   - Predict how many Orthologous Groups (OGs) exist in the set  
   - Assign each gene to its corresponding OG  

   This is achieved by constructing a graph over gene embeddings, learning relationships between genes using a Graph Neural Network (GNN), and applying clustering (MCL) to recover the final groups.
   
## Requirements
The code in this repository was tested with Python 3.13.2. To run the project, you can install the required dependencies using one of the following options:

- **Using pip** (recommended for simplicity):
  ```bash
  pip install -r requirements.txt
  ```
- **Using conda** (for full environment reproducibility):
  ```bash
  conda env create -f environment.yml
  conda activate ML_Project_env
  ```
It is recommended to use a virtual environment (e.g., venv or Conda) to avoid conflicts with existing packages.

## Data

Due to size limitations, the dataset is hosted externally.  
You can download it by clicking on the **"data"** link in this repository, which will take you to a Google Drive folder.

After downloading, make sure to place the files in the correct directory so that the code can access them properly.

## Models

Pretrained models are provided externally.
You can download them by clicking on the **"models"** link in this repository, which will redirect you to a Google Drive folder.
