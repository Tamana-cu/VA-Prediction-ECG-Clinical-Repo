# src/explain.py
# Use Captum integrated gradients for signal saliency and SHAP for clinical features
from captum.attr import IntegratedGradients
# compute attributions over waveform; output visualizations saved to PNG
