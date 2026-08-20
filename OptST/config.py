import argparse
import json
import os


class Config:
    def __init__(self):
        self.DATA_PATH = '../CIKM20-NETE-Datasets/Amazon/Clothing/reviews.pickle'
        self.DATA_PATH_PB = '../CIKM20-NETE-Datasets/Amazon/Clothing/item.json'
        self.INDEX_DIR = '../CIKM20-NETE-Datasets/Amazon/Clothing/3'
        self.PROFILE_VECTORS_PKL = './data/processed/Clothing/local_qwen3/clothing_user_profile_vectors_single_qwen3.pkl'
        self.PROCESSED_CSV = './data/processed/Clothing/clothing_LLM_4reviews.csv'
        self.LOG_DIR = './exp'

        self.PROFILE_DIM = 384
        self.SID_OUTPUT_DIM = 16
        self.EMBED_DIM = 8

        self.DCN_CROSS_LAYERS = 3
        self.DCN_DEEP_LAYERS = [16, 16]
        self.DROPOUT = 0.0

        self.BATCH_SIZE = 2048
        self.LEARNING_RATE = 0.001
        self.WEIGHT_DECAY = 0.00001
        self.NUM_EPOCHS = 100
        self.PATIENCE = 5

        self.DEVICE = 'cuda:0'
        self.SEED = 3407

    def to_dict(self):
        return {
            'data': {
                'data_path': self.DATA_PATH,
                'data_path_pb': self.DATA_PATH_PB,
                'index_dir': self.INDEX_DIR,
                'profile_vectors_pkl': self.PROFILE_VECTORS_PKL,
                'processed_csv': self.PROCESSED_CSV,
            },
            'model': {
                'profile_dim': self.PROFILE_DIM,
                'sid_output_dim': self.SID_OUTPUT_DIM,
                'embed_dim': self.EMBED_DIM,
                'dcn_cross_layers': self.DCN_CROSS_LAYERS,
                'dcn_deep_layers': self.DCN_DEEP_LAYERS,
                'dropout': self.DROPOUT,
            },
            'training': {
                'batch_size': self.BATCH_SIZE,
                'learning_rate': self.LEARNING_RATE,
                'weight_decay': self.WEIGHT_DECAY,
                'num_epochs': self.NUM_EPOCHS,
                'patience': self.PATIENCE,
            },
            'system': {
                'device': self.DEVICE,
                'seed': self.SEED,
            },
            'logging': {
                'log_dir': self.LOG_DIR,
            }
        }

    def save_to_json(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)

    def __repr__(self):
        return f"Config(batch_size={self.BATCH_SIZE}, lr={self.LEARNING_RATE})"


def parse_args():
    default = Config()
    parser = argparse.ArgumentParser(description='OptST training')

    parser.add_argument('--data_path', type=str, default=default.DATA_PATH)
    parser.add_argument('--data_path_pb', type=str, default=default.DATA_PATH_PB)
    parser.add_argument('--index_dir', type=str, default=default.INDEX_DIR)
    parser.add_argument('--profile_vectors_pkl', type=str, default=default.PROFILE_VECTORS_PKL)
    parser.add_argument('--processed_csv', type=str, default=default.PROCESSED_CSV)

    parser.add_argument('--profile_dim', type=int, default=default.PROFILE_DIM)
    parser.add_argument('--sid_output_dim', type=int, default=default.SID_OUTPUT_DIM)
    parser.add_argument('--embed_dim', type=int, default=default.EMBED_DIM)
    parser.add_argument('--dcn_cross_layers', type=int, default=default.DCN_CROSS_LAYERS)
    parser.add_argument('--dcn_deep_layers', type=int, nargs='+', default=default.DCN_DEEP_LAYERS)

    parser.add_argument('--batch_size', type=int, default=default.BATCH_SIZE)
    parser.add_argument('--learning_rate', type=float, default=default.LEARNING_RATE)
    parser.add_argument('--weight_decay', type=float, default=default.WEIGHT_DECAY)
    parser.add_argument('--num_epochs', type=int, default=default.NUM_EPOCHS)
    parser.add_argument('--patience', type=int, default=default.PATIENCE)

    parser.add_argument('--device', type=str, default=default.DEVICE)
    parser.add_argument('--seed', type=int, default=default.SEED)

    parser.add_argument('--log_dir', type=str, default=default.LOG_DIR)

    return parser.parse_args()


def get_config():
    args = parse_args()
    config = Config()

    config.DATA_PATH = args.data_path
    config.DATA_PATH_PB = args.data_path_pb
    config.INDEX_DIR = args.index_dir
    config.PROFILE_VECTORS_PKL = args.profile_vectors_pkl
    config.PROCESSED_CSV = args.processed_csv
    config.PROFILE_DIM = args.profile_dim
    config.SID_OUTPUT_DIM = args.sid_output_dim
    config.EMBED_DIM = args.embed_dim
    config.DCN_CROSS_LAYERS = args.dcn_cross_layers
    config.DCN_DEEP_LAYERS = args.dcn_deep_layers
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate
    config.WEIGHT_DECAY = args.weight_decay
    config.NUM_EPOCHS = args.num_epochs
    config.PATIENCE = args.patience
    config.DEVICE = args.device
    config.SEED = args.seed
    config.LOG_DIR = args.log_dir

    return config
