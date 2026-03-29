## Files Discription
**'bacteria_level_filtered_geneID_to_ogID.py'** 

This script retrieves orthologous group information from the OrthoDB v12 API for the taxonomic level Bacillales (taxonomic ID: 1385).
The script collects gene identifiers belonging to orthologous groups and filters them to include only genes from the families Bacillaceae, Paenibacillaceae, and Alicyclobacillaceae.

**'orthodb_seq.py'**

This script retrieves the amino acid sequences of the genes collected in the previous step.The script queries the OrthoDB v12 FASTA API to download the protein sequences corresponding to each gene.


**'esm_embeddings.py'**

This script converts the protein sequences collected in the previous step into numerical vector representations (embeddings) using the pretrained ESM-2 protein language model.


**'esm_train_model.py'**

This script trains and evaluates machine learning models for the pairwise ortholog prediction task. Using the ESM-2 embeddings generated in the previous step, it computes the cosine distance between pairs of genes and trains two classifiers—Logistic Regression and Random Forest—to predict whether a pair of genes is orthologous. The models are evaluated on validation and test datasets and then saved for later use.

> **Note:** Preprocessed data (including sequences, embeddings, and pair splits) is provided via the external data link. After downloading the data and placing it in the appropriate directory, you can directly run `esm_train_model.py` without executing the previous scripts.
