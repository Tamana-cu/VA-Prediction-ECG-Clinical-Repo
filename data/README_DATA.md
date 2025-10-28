Use PTB-XL for labeled 12-lead ECGs (10s) for initial experiments. PTB-XL is public and good for waveform tasks.

For true near-term VA/SCA labels you need outcome linked datasets (e.g., MUSIC, certain hospital registries, or MIMIC-IV ECG + chart events). These often require controlled access and IRB approval.

Create labels like: event_within_horizon (1 if malignant VA or SCA occurred within X days after ECG). Choose horizon (e.g., 14 days) and justify clinically.

Balance classes using sampling or loss weighting; preserve patient-level splitting.
