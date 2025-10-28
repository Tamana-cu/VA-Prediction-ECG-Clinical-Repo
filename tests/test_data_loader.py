itests/test_data_loader.py
mport os
import numpy as np
import pytest
from src.data_loader import ECGClinicalDataset


@pytest.fixture
def sample_npz(tmp_path):
arr = {
'ecg': np.random.randn(12, 5000).astype('float32'),
'clin': np.random.randn(20).astype('float32'),
'label': np.array(1, dtype='int64')
}
path = tmp_path / 'sample.npz'
np.savez(path, **arr)
return [str(path)]


def test_loader_shape(sample_npz):
ds = ECGClinicalDataset(sample_npz)
ecg, clin, label = ds[0]
assert ecg.shape[0] == 12
assert len(clin) == 20
assert label in [0, 1]
