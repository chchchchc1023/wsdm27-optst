import torch
import pickle
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple


class CTRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, user_profile_vectors: Dict[int, np.ndarray]):
        self.df = df.reset_index(drop=True)
        self.user_profile_vectors = user_profile_vectors

        self.user_ids = self.df['user_id'].values
        self.item_ids = self.df['item_id'].values
        self.labels = self.df['click'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]

        profile_vector = self.user_profile_vectors.get(
            user_id, np.zeros(384)).astype(np.float32)

        return {
            'user_id': user_id,
            'item_id': self.item_ids[idx],
            'profile_vector': torch.from_numpy(profile_vector),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }


def load_data(config) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    print(f"Loading profile vectors from {config.PROFILE_VECTORS_PKL}...")
    with open(config.PROFILE_VECTORS_PKL, 'rb') as f:
        loaded_data = pickle.load(f)

    if isinstance(loaded_data, dict) and 'vectors' in loaded_data and 'uids' in loaded_data:
        vectors = loaded_data['vectors']
        uids = loaded_data['uids']
        user_profile_vectors = {uid: vectors[i] for i, uid in enumerate(uids)}
    elif isinstance(loaded_data, dict):
        user_profile_vectors = loaded_data
    else:
        raise ValueError(f"Unexpected profile vectors format in {config.PROFILE_VECTORS_PKL}")

    print(f"Loading dataset from {config.PROCESSED_CSV}...")
    df = pd.read_csv(config.PROCESSED_CSV)

    num_users = df['user_id'].max() + 1
    num_items = df['item_id'].max() + 1

    print(f"Total samples: {len(df)}")
    print(f"Num users: {num_users}, Num items: {num_items}")

    def load_index_file(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index file not found: {path}")
        with open(path, 'r') as f:
            return [int(x) for x in f.readline().split()]

    print(f"Loading splits from {config.INDEX_DIR}...")
    train_idx = load_index_file(os.path.join(config.INDEX_DIR, 'train.index'))
    val_idx = load_index_file(os.path.join(config.INDEX_DIR, 'validation.index'))
    test_idx = load_index_file(os.path.join(config.INDEX_DIR, 'test.index'))

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    test_df = df.iloc[test_idx]

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    feature_dims = {
        'user_id': num_users,
        'item_id': num_items
    }

    train_dataset = CTRDataset(train_df, user_profile_vectors)
    val_dataset = CTRDataset(val_df, user_profile_vectors)
    test_dataset = CTRDataset(test_df, user_profile_vectors)

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4, pin_memory=True
    )

    train_user_ids = set(train_df['user_id'].unique())

    meta_info = {
        'num_users': num_users,
        'num_items': num_items,
        'feature_dims': feature_dims,
        'user_profile_vectors': user_profile_vectors,
        'train_user_ids': train_user_ids
    }

    return train_loader, val_loader, test_loader, meta_info
