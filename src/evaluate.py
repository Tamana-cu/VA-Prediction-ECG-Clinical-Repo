# src/evaluate.py
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from src.data_loader import ECGClinicalDataset
from src.model import FusionModel


# load checkpoint, compute metrics & calibration
