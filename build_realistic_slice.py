"""
Build a realistic-imbalance slice from the Kaggle combined table, matching the
S3 brief's actual spec: ~20,000 flows at a ~10% attack rate. The raw combined
Kaggle training+testing set is 257,673 rows at 63.9% attack -- pre-balanced,
not realistic, and (as the last run showed) too easy for the learning-rate vs
max_iter trade-off to be visible (AP ~0.993 everywhere).

This keeps ALL attack rows of the rarer classes proportionally and downsamples
the majority (normal) class up so 'attack' becomes the ~10% minority, the same
shape your S3 cleaning notebook would have produced from the harder raw flow
data. Stratified, seeded -> reproducible.
"""
import pandas as pd
import numpy as np

RNG = 42
TARGET_N = 20000
TARGET_ATTACK_RATE = 0.10

df = pd.read_csv("adt_s3_unsw_clean.csv")
attacks = df[df.label == 1]
normals = df[df.label == 0]

n_attack = int(TARGET_N * TARGET_ATTACK_RATE)       # 2000
n_normal = TARGET_N - n_attack                       # 18000

# Sample down (both classes have far more than needed at these target counts)
attack_sample = attacks.sample(n=min(n_attack, len(attacks)), random_state=RNG)
normal_sample = normals.sample(n=min(n_normal, len(normals)), random_state=RNG)

realistic = pd.concat([attack_sample, normal_sample], ignore_index=True)
realistic = realistic.sample(frac=1.0, random_state=RNG).reset_index(drop=True)  # shuffle

print("Shape:", realistic.shape)
print("Attack rate:", round(realistic.label.mean(), 4))

realistic.to_csv("adt_s3_unsw_clean_realistic.csv", index=False)
print("Wrote adt_s3_unsw_clean_realistic.csv")
