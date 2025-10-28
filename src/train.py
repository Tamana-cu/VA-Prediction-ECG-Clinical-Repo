# src/train.py
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from src.data_loader import ECGClinicalDataset
from src.model import FusionModel




def train_loop(cfg):
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load lists of npz files
train_files = open(cfg['train_list']).read().splitlines()
val_files = open(cfg['val_list']).read().splitlines()
train_ds = ECGClinicalDataset(train_files)
val_ds = ECGClinicalDataset(val_files)
trn_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=4)


model = FusionModel(ecg_in=cfg['ecg_leads'], clin_in=cfg['clin_dim']).to(device)
opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.get('pos_weight',1.0))).to(device)


best_auc = 0
for epoch in range(cfg['epochs']):
model.train()
for ecg, clin, y in trn_loader:
ecg = ecg.to(device)
clin = clin.to(device)
y = y.to(device).float()
logits = model(ecg, clin)
loss = loss_fn(logits, y)
opt.zero_grad()
loss.backward()
opt.step()
# val
model.eval()
ys, ps = [], []
with torch.no_grad():
for ecg, clin, y in val_loader:
ecg = ecg.to(device); clin = clin.to(device)
logits = model(ecg, clin)
probs = torch.sigmoid(logits).cpu().numpy()
ys.extend(y.numpy().tolist())
ps.extend(probs.tolist())
auc = roc_auc_score(ys, ps)
ap = average_precision_score(ys, ps)
print(f"Epoch {epoch} val AUC={auc:.4f} AP={ap:.4f}")
if auc>best_auc:
best_auc=auc
torch.save(model.state_dict(), cfg['out_checkpoint'])


if __name__=='__main__':
p = argparse.ArgumentParser()
p.add_argument('--config', required=True)
args = p.parse_args()
cfg = yaml.safe_load(open(args.config))
train_loop(cfg)
