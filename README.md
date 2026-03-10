# Multi-View Speech Emotion Representation

This project implements a Meta-Fusion Variational Autoencoder (MF-VAE) for multi-view speech emotion representation learning.

## Highlights
- 4 view-specific encoders (MFCC, Chroma, Log-Mel, Spectral statistics)
- Meta-fusion gating for adaptive weighted latent fusion
- 4 decoders for multi-view reconstruction
- Losses: reconstruction + consistency + KL + classification
- Fusion strategy comparison: Concatenation, Attention-like, Transformer, Meta-Fusion
- Robustness ablation under view corruption
- Latent space visualization: t-SNE and UMAP

## Repository Contents
- assignment3.ipynb: Main notebook (data loading, preprocessing, training, evaluation)
- fusion_comparison.csv: Fusion strategy performance metrics
- reconstruction_metrics.csv: Reconstruction quality per view
- ablation_weights.csv: Gating weights under clean/corrupt settings
- training_history.csv: Per-epoch training and validation metrics
- training_curves.png: Loss and validation AUC curves
- fusion_comparison_auc.png: Fusion comparison bar plot
- ablation_weight_shift.png: Ablation weight-shift plot
- latent_tsne_meta_fusion.png: t-SNE latent visualization
- latent_umap_meta_fusion.png: UMAP latent visualization
- dimensionality_comparison.csv: Original vs latent dimensionality comparison
- classification_original_vs_latent.csv: Downstream classification comparison
- clustering_original_vs_latent.csv: Downstream clustering comparison

## Experimental Snapshot
- Seed: 42
- Device: CPU
- Sample ratio: 1%
- Samples after filtering: 97
- Train/Test split: 77 / 20
- Input dimensions: 20, 12, 40, 12
- Latent dimension: 32
- Datasets used: CREMA-D, RAVDESS, SAVEE

## Dataset Links
- CREMA-D: https://github.com/CheyneyComputerScience/CREMA-D
- RAVDESS: https://zenodo.org/record/1188976
- SAVEE: https://www.researchgate.net/publication/220147568_The_Surrey_Audio-Visual_Expressed_Emotion_SAVEE_Database

## How To Run
1. Open assignment3.ipynb.
2. Run notebook cells in order.
3. Check generated CSVs and PNGs in the project root (or results folder if exported there).

## Notes
- This version is configured for lightweight experimentation on a very small sampled subset.
- For stronger results, run with larger sampling ratio and GPU training.
