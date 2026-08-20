DATASET_CONFIG = {
    'Clothing': {
        'profile_vectors_pkl': '../data/processed/Clothing/local_qwen3/clothing_user_profile_vectors_single_qwen3.pkl',
        'processed_csv':       '../data/processed/Clothing/clothing_LLM_4reviews.csv',
        'data_path':           '../CIKM20-NETE-Datasets/Amazon/Clothing/reviews.pickle',
        'data_path_pb':        '../CIKM20-NETE-Datasets/Amazon/Clothing/item.json',
        'index_dir':           '../CIKM20-NETE-Datasets/Amazon/Clothing/3',
    },
    'Yelp': {
        'profile_vectors_pkl': '../data/processed/Yelp/local_qwen3/yelp_user_profile_vectors_single_qwen3.pkl',
        'processed_csv':       '../data/processed/Yelp/yelp_LLM_4reviews.csv',
        'data_path':           '../CIKM20-NETE-Datasets/Yelp/reviews.pickle',
        'data_path_pb':        '../CIKM20-NETE-Datasets/Yelp/item.json',
        'index_dir':           '../CIKM20-NETE-Datasets/Yelp/3',
    },
    'Games': {
        'profile_vectors_pkl': '../data/processed/Games/local_qwen3/games_user_profile_vectors_single_qwen3.pkl',
        'processed_csv':       '../data/processed/Games/games_LLM_4reviews.csv',
        'data_path':           '../CIKM20-NETE-Datasets/Amazon/Games/reviews.pickle',
        'data_path_pb':        '../CIKM20-NETE-Datasets/Amazon/Games/item.json',
        'index_dir':           '../CIKM20-NETE-Datasets/Amazon/Games/3',
    },
    'Movies': {
        'profile_vectors_pkl': '../data/processed/Movies/local_qwen3/movies_user_profile_vectors_single_qwen3.pkl',
        'processed_csv':       '../data/processed/Movies/movies_LLM_4reviews.csv',
        'data_path':           '../CIKM20-NETE-Datasets/Amazon/MoviesAndTV/reviews.pickle',
        'data_path_pb':        '../CIKM20-NETE-Datasets/Amazon/MoviesAndTV/item.json',
        'index_dir':           '../CIKM20-NETE-Datasets/Amazon/MoviesAndTV/3',
    },
}

DATASET_CHOICES = list(DATASET_CONFIG.keys())


def apply_dataset_config(config, dataset_name):
    cfg = DATASET_CONFIG[dataset_name]
    config.PROFILE_VECTORS_PKL = cfg['profile_vectors_pkl']
    config.PROCESSED_CSV = cfg['processed_csv']
    config.DATA_PATH = cfg['data_path']
    config.DATA_PATH_PB = cfg['data_path_pb']
    config.INDEX_DIR = cfg['index_dir']
