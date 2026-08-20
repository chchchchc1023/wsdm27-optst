import pickle
import pandas as pd
import os
import torch
import argparse
from transformers import AutoTokenizer
from utils3 import DataLoader0, now_time
from tools import set_seed


DATASET_DEFAULTS = {
    'Movies': {
        'data_path': '../CIKM20-NETE-Datasets/Amazon/MoviesAndTV/reviews.pickle',
        'data_path_pb': '../CIKM20-NETE-Datasets/Amazon/MoviesAndTV/item.json',
        'index_dir': '../CIKM20-NETE-Datasets/Amazon/MoviesAndTV/3',
        'output_csv': '../data/processed/Movies/movies_LLM_4reviews.csv',
    },
    'Games': {
        'data_path': '../CIKM20-NETE-Datasets/Amazon/Games/reviews.pickle',
        'data_path_pb': '../CIKM20-NETE-Datasets/Amazon/Games/item.json',
        'index_dir': '../CIKM20-NETE-Datasets/Amazon/Games/3',
        'output_csv': '../data/processed/Games/games_LLM_4reviews.csv',
    },
    'Clothing': {
        'data_path': '../CIKM20-NETE-Datasets/Amazon/Clothing/reviews.pickle',
        'data_path_pb': '../CIKM20-NETE-Datasets/Amazon/Clothing/item.json',
        'index_dir': '../CIKM20-NETE-Datasets/Amazon/Clothing/3',
        'output_csv': '../data/processed/Clothing/clothing_LLM_4reviews.csv',
    },
    'Yelp': {
        'data_path': '../CIKM20-NETE-Datasets/Yelp/reviews.pickle',
        'data_path_pb': '../CIKM20-NETE-Datasets/Yelp/item.json',
        'index_dir': '../CIKM20-NETE-Datasets/Yelp/3',
        'output_csv': '../data/processed/Yelp/yelp_LLM_4reviews.csv',
    },
}


def _get_first_present(review, keys):
    for key in keys:
        value = review.get(key)
        if value is not None:
            return value
    return None


def _save_ctr_csv(hist_list, output_csv):
    print(now_time() + f"Creating DataFrame with {len(hist_list)} records...")
    hist_df = pd.DataFrame(hist_list, columns=['user_id', 'item_id', 'click'])

    print(now_time() + f'Saving to {output_csv}...')
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    hist_df.to_csv(output_csv, index=False)
    print(now_time() + 'CSV file saved successfully')


def preprocess_yelp(data_path, output_csv):
    print(now_time() + f"Loading Yelp reviews data from {data_path}...")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} does not exist.")
        return

    with open(data_path, 'rb') as f:
        reviews = pickle.load(f)
    print(now_time() + f"Loaded {len(reviews)} reviews")

    user2idx = {}
    item2idx = {}
    hist_list = []

    print(now_time() + f"Processing {len(reviews)} Yelp reviews into CTR format...")
    for i, review in enumerate(reviews):
        if i % 100000 == 0:
            print(now_time() + f"Processed {i}/{len(reviews)} reviews...")

        raw_user = _get_first_present(review, ['user', 'user_id'])
        raw_item = _get_first_present(review, ['item', 'business_id'])
        if raw_user is None or raw_item is None:
            continue

        if raw_user not in user2idx:
            user2idx[raw_user] = len(user2idx)
        if raw_item not in item2idx:
            item2idx[raw_item] = len(item2idx)

        click = 1 if review['rating'] > 3 else 0
        hist_list.append([user2idx[raw_user], item2idx[raw_item], click])

    print(now_time() + f"Mapped {len(user2idx)} users and {len(item2idx)} items")
    _save_ctr_csv(hist_list, output_csv)


def preprocess(dataset_name, data_path, data_path_pb, index_dir, tokenizer_path, output_csv):
    set_seed(3407)
    words = 5

    if dataset_name == 'Yelp':
        preprocess_yelp(data_path, output_csv)
        return

    print(now_time() + f"Loading reviews data from {data_path}...")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} does not exist.")
        return
    
    with open(data_path, 'rb') as f:
        reviews = pickle.load(f)
    print(now_time() + f"Loaded {len(reviews)} reviews")

    print(now_time() + f'Loading tokenizer from {tokenizer_path}...')
    bos = '<bos>'
    eos = '<eos>'
    pad = '<pad>'
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, bos_token=bos, eos_token=eos, pad_token=pad)
    except Exception as e:
        print(f"Warning: Could not load AutoTokenizer from {tokenizer_path}: {e}. Using a placeholder if needed.")
        tokenizer = None

    print(now_time() + 'Loading corpus data and mapping IDs...')
    corpus = DataLoader0(data_path, data_path_pb, index_dir, tokenizer, words)
    print(now_time() + 'Corpus loaded successfully')

    hist_list = []
    print(now_time() + f"Processing {len(reviews)} reviews into CTR format...")
    for i, review in enumerate(reviews):
        if i % 100000 == 0:
            print(now_time() + f"Processed {i}/{len(reviews)} reviews...")

        if review['user'] in corpus.user_dict.entity2idx and review['item'] in corpus.item_dict.entity2idx:
            user_id = corpus.user_dict.entity2idx[review['user']]
            item_id = corpus.item_dict.entity2idx[review['item']]
            
            click = 1 if review['rating'] > 3 else 0
            hist_list.append([user_id, item_id, click])

    _save_ctr_csv(hist_list, output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess raw data to CTR CSV format for OptST')
    parser.add_argument('--dataset', type=str, default='Movies',
                        choices=list(DATASET_DEFAULTS.keys()), help='Dataset name')
    parser.add_argument('--data_path', type=str, default=None, help='Path to reviews.pickle')
    parser.add_argument('--data_path_pb', type=str, default=None, help='Path to item.json')
    parser.add_argument('--index_dir', type=str, default=None, help='Path to index directory')
    parser.add_argument('--tokenizer_path', type=str, default='../models/Qwen3-8B', help='Path to tokenizer')
    parser.add_argument('--output_csv', type=str, default=None, help='Output CSV path')
    
    args = parser.parse_args()
    defaults = DATASET_DEFAULTS[args.dataset]

    preprocess(
        args.dataset,
        args.data_path or defaults['data_path'],
        args.data_path_pb or defaults['data_path_pb'],
        args.index_dir or defaults['index_dir'],
        args.tokenizer_path,
        args.output_csv or defaults['output_csv'])
