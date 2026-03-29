## Files Discription

**'sample_gene_sets.py'**

This script generates the synthetic gene sets used for the set-level orthologous group inference task. It loads the gene-to-OG mapping and ESM-2 embeddings, filters out genes without embeddings, and splits the OGs into development and test sets to prevent leakage.
Gene sets of different sizes are then created by sampling genes from multiple OGs, with a variable number of genes per OG. This produces heterogeneous sets containing both large groups and singletons, while limiting dominance of large OGs.
The script outputs a split file and two CSV files (dev_sets.csv, test_sets.csv), where each row represents a gene set stored as a comma-separated list in the gene_ids column.

**'set_gnn_pipeline.py'**

This script implements the main learning and inference pipeline for set-level orthologous group prediction . It first loads the generated gene sets, embeddings, and OG mappings, and constructs supervised edge labels within each set based on the known OG partition. These edges include both positive pairs (same OG) and negative pairs (different OG), with additional hard negatives sampled using k-nearest neighbors.
A Graph Neural Network (GNN) encoder is then trained on each set using a kNN graph built from the embeddings. The model learns node representations and predicts pairwise relationships between genes through an edge scoring network. Training is performed using a combined loss consisting of binary cross-entropy for edge classification and a triplet loss that encourages genes from the same OG to be close in the embedding space.
At inference time, the learned edge scores are used to construct a weighted graph, which is clustered using Markov Clustering (MCL) to recover the final OG partition of each set. The script also includes evaluation procedures that measure clustering quality using metrics such as OG count error, singleton precision/recall, coverage, and IoU.
