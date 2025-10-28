# src/data_loader.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ECGClinicalDataset(Dataset):
"""Dataset that returns (ecg_tensor, clinical_tensor, label)
Assumes preprocessed .npz files with keys: 'ecg', 'clin', 'label', 'patient_id'
ecg: (n_leads, n_samples)
clin: 1D vector of clinical features
label: 0/1
"""
def __init__(self, files, transform=None):
self.files = files
self.transform = transform


def __len__(self):
return len(self.files)


def __getitem__(self, idx):
arr = np.load(self.files[idx])
ecg = arr['ecg'].astype('float32')
clin = arr['clin'].astype('float32')
label = np.int64(arr['label'])
if self.transform:
ecg = self.transform(ecg)
return torch.from_numpy(ecg), torch.from_numpy(clin), torch.tensor(label)
