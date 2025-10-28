# src/model.py
import torch
import torch.nn as nn


class ECGEncoder(nn.Module):
def __init__(self, in_channels=12, out_dim=128):
super().__init__()
self.net = nn.Sequential(
nn.Conv1d(in_channels,32,kernel_size=15,stride=1,padding=7),
nn.BatchNorm1d(32),
nn.ReLU(),
nn.MaxPool1d(2),
nn.Conv1d(32,64,kernel_size=15,padding=7),
nn.BatchNorm1d(64),
nn.ReLU(),
nn.AdaptiveAvgPool1d(1),
nn.Flatten(),
nn.Linear(64, out_dim),
nn.ReLU()
)
def forward(self,x):
return self.net(x)


class ClinicalEncoder(nn.Module):
def __init__(self, in_dim, out_dim=64):
super().__init__()
self.net = nn.Sequential(
nn.Linear(in_dim,128),
nn.ReLU(),
nn.Linear(128,out_dim),
nn.ReLU()
)
def forward(self,x):
return self.net(x)


class FusionModel(nn.Module):
def __init__(self, ecg_in=12, clin_in=20):
super().__init__()
self.ecg_enc = ECGEncoder(in_channels=ecg_in, out_dim=128)
self.clin_enc = ClinicalEncoder(in_dim=clin_in, out_dim=64)
self.head = nn.Sequential(
nn.Linear(128+64,64),
nn.ReLU(),
nn.Dropout(0.3),
nn.Linear(64,1)
)
def forward(self, ecg, clin):
# ecg: (B, leads, samples)
e = self.ecg_enc(ecg)
c = self.clin_enc(clin)
x = torch.cat([e,c], dim=1)
return self.head(x).squeeze(1)
