"""
@author AFelixLiu
@date 2024 12月 18
"""

from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def generate_data(smiles, labels, radius=11, splits=None, Nmin=4):
    if splits is None:
        splits = [0.7, 0.15, 0.15]
    num_samples = len(smiles)
    indexes = np.random.permutation(num_samples)
    train_indexes = indexes[:int(len(indexes) * splits[0])]
    val_indexes = indexes[int(len(indexes) * splits[0]):int(len(indexes) * (splits[0] + splits[1]))]
    test_indexes = indexes[int(len(indexes) * (splits[0] + splits[1])):]

    morganfps = np.empty(num_samples, dtype=object)
    features_counter = defaultdict(int)

    for idx in range(0, num_samples):
        curr_smiles = smiles[idx]
        mol = Chem.MolFromSmiles(curr_smiles)
        fp = AllChem.GetMorganFingerprint(mol, radius)
        morganfps[idx] = fp
        if idx in train_indexes:
            for feature in [*fp.GetNonzeroElements()]:
                features_counter[feature] += 1

    morganfp_features = np.empty(len(features_counter.keys()), dtype=object)
    cnt = 0
    for feature in features_counter.keys():
        if features_counter[feature] >= Nmin:
            morganfp_features[cnt] = feature
            cnt += 1
    morganfp_features = list(morganfp_features[:cnt])

    num_features = len(morganfp_features)
    smiles2fp = np.zeros((len(smiles), num_features))
    for idx in range(0, len(smiles)):
        for feature in [*morganfps[idx].GetNonzeroElements()]:
            if feature in morganfp_features:
                pos = morganfp_features.index(feature)
                smiles2fp[idx][pos] = morganfps[idx][feature]

    result = {"smiles_train": smiles[train_indexes],
              "smiles_val": smiles[val_indexes],
              "smiles_test": smiles[test_indexes],
              "X_train": smiles2fp[train_indexes],
              "X_valid": smiles2fp[val_indexes],
              "X_test": smiles2fp[test_indexes],
              "Y_train": labels[train_indexes],
              "Y_valid": labels[val_indexes],
              "Y_test": labels[test_indexes],
              "morganfp_features": morganfp_features,
              "radius": radius,
              "Nmin": Nmin}

    return result
