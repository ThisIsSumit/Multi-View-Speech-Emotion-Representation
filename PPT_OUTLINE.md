# PPT Outline (Template-Aligned)

## Slide 1: Title
- Meta-Fusion Variational Autoencoder for Multi-View Speech Emotion Representation
- Name, course, date

## Slide 2: Problem Statement
- Why speech emotion recognition is hard
- Why single-view representations are insufficient
- Objective: robust fused latent representation

## Slide 3: Dataset and Setup
- Datasets: CREMA-D, RAVDESS, SAVEE
- Classes: angry, calm, disgust, fear, happy, neutral, sad, surprise
- Current run config: 1% sampling, CPU, 78/20 split

## Slide 4: Feature Views and Preprocessing
- Four view dimensions: 20, 12, 40, 12
- Scaling/normalization pipeline
- Split strategy and class distribution notes

## Slide 5: Proposed Architecture
- 4 encoders -> latent variables -> meta-gating -> fused latent -> 4 decoders
- Optional supervised head
- Include architecture diagram

## Slide 6: Loss Components
- Reconstruction (L_rec)
- Consistency (L_cons)
- KL (L_kl)
- Classification (L_cls)
- Total weighted objective

## Slide 7: Fusion Baselines
- Concatenation
- Attention-like
- Transformer
- Meta-Fusion (proposed)

## Slide 8: Training Dynamics
- Plot: training_curves.png
- Explain trend in total and component losses

## Slide 9: Reconstruction Quality
- Table: reconstruction_metrics.csv
- Highlight best view and discuss metric behavior (MAPE caveat)

## Slide 10: Fusion Performance
- Table/plot: fusion_comparison.csv and fusion_comparison_auc.png
- Explain current low-data limitation

## Slide 11: Ablation and Robustness
- Plot: ablation_weight_shift.png
- Explain weight adaptation under corrupted V1

## Slide 12: Latent Space Visualization
- t-SNE (and UMAP if available) from results/
- Compare cluster compactness vs single-view baseline

## Slide 13: Original vs Latent Benchmarks
- Dimensionality reduction table
- Classification and clustering comparison
- Optional regression results

## Slide 14: Cross-Dataset Generalization Plan
- Domain-adversarial branch (GRL)
- Progressive pretrain/fine-tune
- Per-dataset latent normalization

## Slide 15: Why Meta-Fusion
- Mechanism-level argument: adaptive weighting under corruption
- Practical benefit over concatenation in non-stationary multi-view settings

## Slide 16: Limitations and Future Work
- Small sampled run currently underpowered
- Multi-GPU + full data training target
- Better calibration and domain adaptation

## Slide 17: Conclusion
- Summary of contribution
- Key findings and next steps

## Slide 18: References
- IEEE-format references used in report
