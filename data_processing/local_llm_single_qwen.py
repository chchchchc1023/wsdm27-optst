import pandas as pd
import pickle
import numpy as np
import json
import time
import os
import torch
import argparse
from collections import defaultdict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from utils3 import DataLoader0
from tools import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM

LOCAL_MODEL_PATH = '../models/Qwen3-8B/'
BATCH_SIZE = 30  # Increase this if GPU memory allows.
MAX_NEW_TOKENS = 150  # Maximum generated tokens per profile.


def parse_args():
    parser = argparse.ArgumentParser(description='Generate user profiles using local LLM')
    parser.add_argument('--dataset', type=str, default='Games', 
                        choices=['Movies', 'Games', 'Yelp', 'Clothing'],
                        help='Dataset to process (default: Games)')
    return parser.parse_args()

args = parse_args()
TARGET_DATASET = args.dataset

DATASET_CONFIG = {
    'Movies': {
        'root': '../CIKM20-NETE-Datasets/Amazon/MoviesAndTV',
        'base_csv': '../data/processed/Movies/movies_LLM_4reviews.csv',
        'ckpt_file': '../data/processed/Movies/local_qwen3/amazon_movie_user_profiles_single_qwen3.jsonl',
    },
    'Games': {
        'root': '../CIKM20-NETE-Datasets/Amazon/Games',
        'base_csv': '../data/processed/Games/games_LLM_4reviews.csv',
        'ckpt_file': '../data/processed/Games/local_qwen3/amazon_games_user_profiles_single_qwen3.jsonl',
    },
    'Clothing': {
        'root': '../CIKM20-NETE-Datasets/Amazon/Clothing',
        'base_csv': '../data/processed/Clothing/clothing_LLM_4reviews.csv',
        'ckpt_file': '../data/processed/Clothing/local_qwen3/amazon_clothing_user_profiles_single_qwen3.jsonl',
    },
    'Yelp': {
        'root': '../CIKM20-NETE-Datasets/Yelp',
        'base_csv': '../data/processed/Yelp/yelp_LLM_4reviews.csv',
        'ckpt_file': '../data/processed/Yelp/local_qwen3/yelp_user_profiles_single_qwen3.jsonl',
    }
}

cfg = DATASET_CONFIG[TARGET_DATASET]

if TARGET_DATASET == 'Yelp':
    DATA_PATH_REVIEWS = os.path.join(cfg['root'], 'reviews.pickle')
    DATA_PATH_ITEM = os.path.join(cfg['root'], 'item.json')
    DATA_PATH_USER = os.path.join(cfg['root'], 'user.json')
    INDEX_DIR = os.path.join(cfg['root'], '3')
    BASE_CSV_PATH = cfg['base_csv']
else:
    DATA_PATH = os.path.join(cfg['root'], 'reviews.pickle')
    DATA_PATH_PB = os.path.join(cfg['root'], 'item.json')
    INDEX_DIR = os.path.join(cfg['root'], '3')
    BASE_CSV_PATH = cfg['base_csv']

CHECKPOINT_FILE = cfg['ckpt_file']
OUTPUT_VECTOR_PKL = f"../data/processed/{TARGET_DATASET}/local_qwen3/{TARGET_DATASET.lower()}_user_profile_vectors_single_qwen3.pkl"

set_seed(3407)

class EntityDictionary:
    def __init__(self):
        self.idx2entity = []
        self.entity2idx = {}

    def add_entity(self, e):
        if e not in self.entity2idx:
            self.entity2idx[e] = len(self.idx2entity)
            self.idx2entity.append(e)


def load_train_indices(index_dir):
    file_path = os.path.join(index_dir, 'train.index')
    print(f"Loading train indices from {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Training split not found: {file_path}. "
            "Profile generation requires train.index and will not fall back "
            "to the full interaction data.")
    with open(file_path, 'r') as f:
        line = f.readline()
    if not line.strip():
        raise ValueError(f"Training split is empty: {file_path}")
    return {int(x) for x in line.split()}


def get_review_user_item(review):
    raw_user = review.get('user', review.get('user_id'))
    raw_item = review.get('item', review.get('business_id'))
    return raw_user, raw_item


def prepare_user_history():
    if TARGET_DATASET == 'Yelp':
        return prepare_yelp_data()
    else:
        return prepare_amazon_data()


def prepare_amazon_data():
    print(f"[{TARGET_DATASET}] Initializing DataLoader...")

    class DummyTokenizer:
        def __call__(self, text, **kwargs): return {'input_ids': []}
        def decode(self, ids, **kwargs): return ""

    corpus = DataLoader0(DATA_PATH, DATA_PATH_PB, INDEX_DIR,
                         DummyTokenizer(), seq_len=10)
    print("Loading raw reviews...")
    reviews = pickle.load(open(DATA_PATH, 'rb'))
    train_indices = load_train_indices(INDEX_DIR)

    print("Aggregating User History (STRICT MODE: Training Data Only)...")
    user_data_store = defaultdict(
        lambda: {'reviews': [], 'brands': [], 'prices': []})

    for idx, review in enumerate(reviews):
        if train_indices is None or idx in train_indices:
            u_idx = corpus.user_dict.entity2idx[review['user']]
            user_data_store[u_idx]['reviews'].append(review['template'][2])
            raw_item_id = review['item']
            if raw_item_id in corpus.brand_dict:
                user_data_store[u_idx]['brands'].append(
                    corpus.brand_dict[raw_item_id])
            if raw_item_id in corpus.price_dict:
                user_data_store[u_idx]['prices'].append(
                    corpus.price_dict[raw_item_id])
    return user_data_store, corpus


def prepare_yelp_data():
    print("Step 1: Loading Raw Data...")
    if not os.path.exists(DATA_PATH_REVIEWS):
        raise FileNotFoundError(f"Not found: {DATA_PATH_REVIEWS}")
    reviews = pickle.load(open(DATA_PATH_REVIEWS, 'rb'))
    user_dict = EntityDictionary()
    item_dict = EntityDictionary()
    for review in reviews:
        raw_user, raw_item = get_review_user_item(review)
        if raw_user is None or raw_item is None:
            continue
        user_dict.add_entity(raw_user)
        item_dict.add_entity(raw_item)
    print("Loading Item Categories...")
    item_hobby_map = {}
    with open(DATA_PATH_ITEM, 'r', encoding='utf-8') as f:
        item_data = json.load(f)
        for item in item_data:
            iid = item.get('item', item.get('business_id'))
            if not iid:
                continue
            categories = item.get('categories', "")
            cat_str = ", ".join(categories) if isinstance(
                categories, list) else str(categories)
            item_hobby_map[iid] = cat_str.split(
                ', ')[-1].strip() if ',' in cat_str else cat_str

    print("Loading User Stats...")
    user_bg_map = {}
    with open(DATA_PATH_USER, 'r', encoding='utf-8') as f:
        user_data = json.load(f)
        for u in user_data:
            uid = u.get('user', u.get('user_id'))
            if not uid:
                continue
            user_bg_map[uid] = {'review_count': u.get('review_count', 0), 'useful': u.get(
                'useful', 0), 'funny': u.get('funny', 0), 'cool': u.get('cool', 0)}

    print("Aggregating User History...")
    user_agg = defaultdict(lambda: {'reviews': [], 'hobbies': [], 'bg': {}})
    train_indices = load_train_indices(INDEX_DIR)

    for idx, review in enumerate(reviews):
        if train_indices is None or idx in train_indices:
            raw_u, raw_i = get_review_user_item(review)
            if raw_u is None or raw_i is None:
                continue
            u_idx = user_dict.entity2idx[raw_u]
            user_agg[u_idx]['reviews'].append(review['template'][2])
            if raw_i in item_hobby_map:
                user_agg[u_idx]['hobbies'].append(item_hobby_map[raw_i])
            if not user_agg[u_idx]['bg'] and raw_u in user_bg_map:
                user_agg[u_idx]['bg'] = user_bg_map[raw_u]
    return user_agg, reviews, user_dict, item_dict, train_indices

def load_model_and_tokenizer():
    print(f"Loading Qwen3-8B from {LOCAL_MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_PATH,
        local_files_only=True
    )

    # Qwen3 usually has a pad token; fall back to EOS if needed.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model.eval()
    return tokenizer, model


def batch_generate(prompts, tokenizer, model):
    if not prompts:
        return []

    # Use the chat template and disable Qwen3 thinking mode.
    texts = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        )
        texts.append(text)

    inputs = tokenizer(texts, return_tensors='pt', padding=True,
                       truncation=True, max_length=1024).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.5,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    input_len = inputs.input_ids.shape[1]
    generated_tokens = outputs[:, input_len:]
    decoded = tokenizer.batch_decode(
        generated_tokens, skip_special_tokens=True)
    return [text.strip() for text in decoded]


def construct_prompt(u_idx, user_store):
    if TARGET_DATASET == 'Yelp':
        return construct_yelp_prompt(u_idx, user_store)
    else:
        return construct_amazon_prompt(u_idx, user_store)


def construct_amazon_prompt(u_idx, user_store):
    data = user_store[u_idx]
    recent_reviews = "; ".join(data['reviews'][-5:])
    recent_brands = ", ".join(list(set(data['brands']))[-10:]) or "Various"
    recent_prices = ", ".join(data['prices'][-10:]) or "Standard"

    prompt = ""
    if TARGET_DATASET == 'Movies':
        prompt = f"""
        Analyze the following user's movie viewing history:
        - Preferred Genres/Studios: {recent_brands}
        - Consumption Level (Price): {recent_prices}
        - Recent Reviews: "{recent_reviews}"
        Task: Create a concise user profile (under 50 words) describing their movie taste (e.g., likes horror, prefers drama) and spending habits.
        Profile:
        """
    elif TARGET_DATASET == 'Games':
        prompt = f"""
        Analyze the following user's gaming history:
        - Preferred Studios/Genres: {recent_brands}
        - Consumption Level (Price): {recent_prices}
        - Recent Reviews: "{recent_reviews}"
        Task: Create a concise user profile (under 50 words) describing their gaming preferences (e.g., likes RPG, prefers FPS) and spending habits.
        Profile:
        """
    elif TARGET_DATASET == 'Clothing':
        prompt = f"""
        Analyze the following user's clothing purchase history:
        - Preferred Brands/Styles: {recent_brands}
        - Consumption Level (Price): {recent_prices}
        - Recent Reviews: "{recent_reviews}"
        Task: Create a concise user profile (under 50 words) describing their fashion style (e.g., likes casual, prefers formal) and spending habits.
        Profile:
        """
    return prompt.strip()


def construct_yelp_prompt(u_idx, user_agg):
    data = user_agg[u_idx]
    reviews = "; ".join(data['reviews'][-5:])
    categories = ", ".join(list(set(data['hobbies']))[-10:]) or "Various"
    bg = data['bg']
    stats = (f"Review Count: {bg.get('review_count', 0)}, Useful Votes: {bg.get('useful', 0)}, "
             f"Cool Votes: {bg.get('cool', 0)}, Funny Votes: {bg.get('funny', 0)}")
    prompt = f"""
    Analyze the following user's dining and service history:
    - Preferred Categories: {categories}
    - Community Influence (Stats): {stats}
    - Recent Reviews: "{reviews}"
    Task: Create a concise user profile (under 50 words) describing their dining taste and activity level.
    Profile:
    """
    return prompt.strip()

def main():
    print(
        f"=== Starting Process for: {TARGET_DATASET} (Single Mode | Local Qwen3-8B) ===")

    tokenizer, model = load_model_and_tokenizer()

    if TARGET_DATASET == 'Yelp':
        user_agg, reviews, user_dict, item_dict, train_indices = prepare_user_history()
        unique_users = sorted(user_agg.keys())
        user_data_src = user_agg
    else:
        user_data_src, corpus = prepare_user_history()
        if not os.path.exists(BASE_CSV_PATH):
            raise FileNotFoundError(f"Base CSV not found at {BASE_CSV_PATH}.")
        base_df = pd.read_csv(BASE_CSV_PATH)
        unique_users = sorted(base_df['user_id'].unique())

    print(f"Total unique users to process: {len(unique_users)}")

    finished_results = {}
    if os.path.exists(CHECKPOINT_FILE):
        print(f"Loading existing profiles from {CHECKPOINT_FILE}...")
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    finished_results[d['user_id']] = d['profile']
                except:
                    continue
        print(f"Loaded {len(finished_results)} profiles.")

    users_to_process = [u for u in unique_users if u not in finished_results]
    print(f"Remaining users to process: {len(users_to_process)}")

    if users_to_process:
        os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
        for i in tqdm(range(0, len(users_to_process), BATCH_SIZE), desc="Qwen3 Generation"):
            batch_uids = users_to_process[i: i + BATCH_SIZE]

            prompts = [construct_prompt(uid, user_data_src)
                       for uid in batch_uids]

            try:
                results = batch_generate(prompts, tokenizer, model)

                with open(CHECKPOINT_FILE, 'a', encoding='utf-8') as f:
                    for j, uid in enumerate(batch_uids):
                        res = {"user_id": int(uid), "profile": results[j]}
                        f.write(json.dumps(res, ensure_ascii=False) + "\n")
                        finished_results[uid] = results[j]
            except RuntimeError as e:
                print(f"OOM or Error in batch {i}: {e}")
                torch.cuda.empty_cache()
                continue
    else:
        print("All profiles ready.")

    print("Encoding profiles to dense vectors...")

    del model
    del tokenizer
    torch.cuda.empty_cache()

    ordered_profiles = []
    for u in unique_users:
        if u in finished_results:
            ordered_profiles.append(finished_results[u])
        else:
            ordered_profiles.append(f"General {TARGET_DATASET} user")

    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    vectors = embed_model.encode(
        ordered_profiles, batch_size=256, show_progress_bar=True)
    vectors = np.array(vectors)

    os.makedirs(os.path.dirname(OUTPUT_VECTOR_PKL), exist_ok=True)
    print(f"Saving vector embeddings to {OUTPUT_VECTOR_PKL}...")
    user_vec_dict = {int(u): vectors[i] for i, u in enumerate(unique_users)}

    with open(OUTPUT_VECTOR_PKL, 'wb') as f:
        pickle.dump(user_vec_dict, f)

    print(f"Vector embeddings saved to: {OUTPUT_VECTOR_PKL}")
    print("Process Complete (Local Qwen3-8B Single Profile).")


if __name__ == "__main__":
    main()
