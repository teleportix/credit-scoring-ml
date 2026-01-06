The Logistic Regression baseline achieves a ROC-AUC of approximately 0.74, indicating that the selected and engineered features provide meaningful discriminatory power between defaulters and non-defaulters.

Threshold analysis demonstrates the expected recall–precision trade-off and allows decision thresholds to be selected according to business risk tolerance rather than relying on a fixed default cutoff.

Feature ablation experiments show that even features with small marginal coefficients contribute to overall model stability and ranking performance, and are therefore retained to improve robustness rather than being removed aggressively