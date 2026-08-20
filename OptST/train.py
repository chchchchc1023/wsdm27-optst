import os
import sys
import logging
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from config import get_config
from model import OptSTModel
from dataset import load_data
from metrics import evaluate_model
from tools import set_seed, set_logger
from datasets import DATASET_CHOICES, apply_dataset_config


DEFAULT_SEED = 3407


def _auto_detect_from_meta(pretrained_dir, field, override):
    if override is not None:
        return override
    meta_path = os.path.join(pretrained_dir, 'meta.pt')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"meta.pt not found in {pretrained_dir}; pass --{field} explicitly")
    meta = torch.load(meta_path, map_location='cpu')
    if field not in meta:
        raise KeyError(
            f"'{field}' missing from {meta_path}; pass --{field} explicitly")
    return meta[field]


def get_optst_config():
    original_argv = list(sys.argv)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', type=str, default=None,
                        choices=DATASET_CHOICES)
    parser.add_argument('--pretrained_dir', type=str, required=True,
                        help='Stage 1 output directory (encoder/codebooks; optional decoder.pt)')
    parser.add_argument('--max_layers', type=int, default=8)
    parser.add_argument('--latent_dim', type=int, default=None,
                        help='Defaults to meta.pt value in --pretrained_dir')
    parser.add_argument('--encoder_hidden', type=int, default=None,
                        help='Defaults to meta.pt value in --pretrained_dir')

    parser.add_argument('--depth_lambda', type=float, default=0.0)
    parser.add_argument('--anchor_lambda', type=float, default=0.0)
    parser.add_argument('--anchor_m', type=int, default=5)
    parser.add_argument('--anchor_epsilon', type=float, default=0.0)

    parser.add_argument('--tau_depth_init', type=float, default=1.0)
    parser.add_argument('--tau_min', type=float, default=0.1)
    parser.add_argument('--tau_decay', type=float, default=0.95)
    parser.add_argument('--gate_hidden_dim', type=int, default=16)
    parser.add_argument('--gate_init_bias', type=float, default=1.5)

    parser.add_argument('--finetune_encoder',
                        action=argparse.BooleanOptionalAction, default=True,
                        help='Unfreeze encoder/decoder in Stage 2 '
                             '(default: True; pass --no-finetune_encoder to freeze)')
    parser.add_argument('--encoder_lr_scale', type=float, default=0.1)
    parser.add_argument('--encoder_wd', type=float, default=None,
                        help='Adam wd for encoder+decoder. Default: tie to --weight_decay.')

    parser.add_argument('--lora_rank', type=int, default=4)
    parser.add_argument('--lora_init_scale', type=float, default=0.01)
    parser.add_argument('--lora_wd', type=float, default=1e-4,
                        help='Adam wd on LoRA W_down/W_up (independent of --weight_decay).')
    parser.add_argument('--lora_lr_mult', type=float, default=1.0)

    parser.add_argument('--ste_temperature', type=float, default=1.0)
    parser.add_argument('--emb_wd', type=float, default=None,
                        help='Adam wd on per-layer id Embedding tables. Default: tie to --weight_decay.')
    parser.add_argument('--emb_lr_mult', type=float, default=1.0)

    parser.add_argument('--gate_lr_mult', type=float, default=1.0)
    parser.add_argument('--gate_wd', type=float, default=None,
                        help='Adam wd on gate parameters. Default: tie to --weight_decay.')

    optst_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    config = get_config()

    if optst_args.dataset is not None:
        apply_dataset_config(config, optst_args.dataset)

    config.PRETRAINED_DIR = optst_args.pretrained_dir
    config.MAX_LAYERS = optst_args.max_layers
    config.LATENT_DIM = _auto_detect_from_meta(
        optst_args.pretrained_dir, 'latent_dim', optst_args.latent_dim)
    config.ENCODER_HIDDEN_DIM = _auto_detect_from_meta(
        optst_args.pretrained_dir, 'encoder_hidden', optst_args.encoder_hidden)
    config.DEPTH_LAMBDA = optst_args.depth_lambda
    config.ANCHOR_LAMBDA = optst_args.anchor_lambda
    config.ANCHOR_M = optst_args.anchor_m
    config.ANCHOR_EPSILON = optst_args.anchor_epsilon
    config.TAU_DEPTH_INIT = optst_args.tau_depth_init
    config.TAU_MIN = optst_args.tau_min
    config.TAU_DECAY = optst_args.tau_decay
    config.GATE_HIDDEN_DIM = optst_args.gate_hidden_dim
    config.GATE_INIT_BIAS = optst_args.gate_init_bias
    config.FINETUNE_ENCODER = optst_args.finetune_encoder
    config.ENCODER_LR_SCALE = optst_args.encoder_lr_scale
    config.ENCODER_WD = optst_args.encoder_wd

    config.LORA_RANK = optst_args.lora_rank
    config.LORA_INIT_SCALE = optst_args.lora_init_scale
    config.LORA_WD = optst_args.lora_wd
    config.LORA_LR_MULT = optst_args.lora_lr_mult
    config.STE_TEMPERATURE = optst_args.ste_temperature
    config.EMB_WD = optst_args.emb_wd
    config.EMB_LR_MULT = optst_args.emb_lr_mult

    config.GATE_LR_MULT = optst_args.gate_lr_mult
    config.GATE_WD = optst_args.gate_wd

    seed_explicit = any(
        a == '--seed' or a.startswith('--seed=') for a in original_argv)
    if not seed_explicit:
        config.SEED = DEFAULT_SEED
    return config


def anneal_temperature(epoch, tau_init, tau_min, decay):
    return max(tau_min, tau_init * (decay ** epoch))


def train_epoch(model, train_loader, optimizer, device, epoch, config):
    model.train()
    sums = {'loss': 0, 'ctr': 0, 'depth': 0, 'anchor': 0}
    start = time.time()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        profile_vectors = batch['profile_vector'].to(device)
        user_ids = batch['user_id'].to(device)
        item_ids = batch['item_id'].to(device)
        labels = batch['label'].to(device).float().view(-1, 1)
        features = {'user_id': user_ids, 'item_id': item_ids}

        predictions = model(profile_vectors, features)
        ctr_loss = nn.BCELoss()(predictions, labels)
        depth_loss = model.compute_depth_loss()
        anchor_loss = model.compute_anchor_loss()

        loss = (ctr_loss
                + config.DEPTH_LAMBDA * depth_loss
                + config.ANCHOR_LAMBDA * anchor_loss)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        depth_loss_raw = depth_loss.item() * model.max_layers

        sums['loss'] += loss.item()
        sums['ctr'] += ctr_loss.item()
        sums['depth'] += depth_loss_raw
        sums['anchor'] += (anchor_loss.item()
                           if isinstance(anchor_loss, torch.Tensor) else anchor_loss)

        pbar.set_postfix({
            'ctr': f'{ctr_loss.item():.4f}',
            'dep': f'{depth_loss_raw:.2f}',
        })

    elapsed = time.time() - start
    n = len(train_loader)
    avgs = {k: v / n for k, v in sums.items()}
    return avgs, elapsed


def reset_logging():
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()





def run_training(config):
    max_layers = config.MAX_LAYERS

    print(config)
    print(f"[OptST Stage 2] layers={max_layers}, "
          f"latent_dim={config.LATENT_DIM}, "
          f"lora_rank={config.LORA_RANK}, "
          f"ste_τ={config.STE_TEMPERATURE}")
    print(f"  depth_lambda={config.DEPTH_LAMBDA}, "
          f"anchor_lambda={config.ANCHOR_LAMBDA}, "
          f"gate_bias={config.GATE_INIT_BIAS}")
    print(f"  tau_depth_init={config.TAU_DEPTH_INIT}, "
          f"tau_min={config.TAU_MIN}, decay={config.TAU_DECAY}")
    print(f"  finetune_encoder={config.FINETUNE_ENCODER}, "
          f"lora_wd={config.LORA_WD}, "
          f"emb_wd={config.EMB_WD if config.EMB_WD is not None else f'<tie wd={config.WEIGHT_DECAY}>'}")
    print(f"  seed={config.SEED}")

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    exp_name = f"OptST_{max_layers}L_{timestamp}"
    reset_logging()
    exp_dir = set_logger(log_dir=config.LOG_DIR, exp_name=exp_name)
    set_seed(config.SEED)
    config.save_to_json(f"{exp_dir}/config.json")

    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    config.DEVICE = str(device)
    print(f"Using device: {device}")
    logging.info(f"Using device: {device}")

    load_start = time.time()
    train_loader, val_loader, test_loader, meta_info = load_data(config)
    print(f"Data load time: {time.time() - load_start:.2f}s")
    print(f"Train samples: {len(train_loader.dataset)}")

    model = OptSTModel(
        profile_dim=config.PROFILE_DIM,
        latent_dim=config.LATENT_DIM,
        sid_output_dim=config.SID_OUTPUT_DIM,
        feature_dims=meta_info['feature_dims'],
        config=config,
        max_layers=max_layers,
        anchor_m=config.ANCHOR_M,
        anchor_epsilon=config.ANCHOR_EPSILON,
    ).to(device)

    model.load_pretrained(config.PRETRAINED_DIR, device)

    if not config.FINETUNE_ENCODER:
        model.freeze_encoder()
        print("Encoder/decoder frozen")
    else:
        print("Encoder/decoder will be fine-tuned")

    num_users = meta_info['num_users']
    num_items = meta_info['num_items']
    train_ds = train_loader.dataset
    u_arr = np.asarray(train_ds.user_ids)
    i_arr = np.asarray(train_ds.item_ids)
    user_freq = torch.zeros(num_users, device=device)
    item_freq = torch.zeros(num_items, device=device)
    u_unique, u_counts = np.unique(u_arr, return_counts=True)
    i_unique, i_counts = np.unique(i_arr, return_counts=True)
    u_keep = u_unique < num_users
    i_keep = i_unique < num_items
    user_freq[torch.from_numpy(u_unique[u_keep]).long().to(device)] = \
        torch.from_numpy(u_counts[u_keep]).float().to(device)
    item_freq[torch.from_numpy(i_unique[i_keep]).long().to(device)] = \
        torch.from_numpy(i_counts[i_keep]).float().to(device)
    model.set_user_item_freqs(user_freq, item_freq)

    print(model)
    logging.info(str(model))

    codebook_str = ', '.join(
        [f"L{i+1}={model._codebook_sizes[i]}"
         for i in range(max_layers)])
    print(f"Frozen codebooks (latent_dim={config.LATENT_DIM}): {codebook_str}")
    logging.info(f"Frozen codebooks: {codebook_str}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    lora_params = sum(p.numel() for p in model.lora_adapters.parameters())
    emb_params = sum(p.numel() for p in model.code_embeddings.parameters())
    print(f"Parameters: total={total_params}, trainable={trainable_params}, "
          f"lora={lora_params}, emb_tables={emb_params}")
    logging.info(f"Parameters: total={total_params}, "
                 f"trainable={trainable_params}, "
                 f"lora={lora_params}, emb_tables={emb_params}")

    enc_lr_scale = float(getattr(config, 'ENCODER_LR_SCALE', 0.1))
    gate_lr_mult = float(getattr(config, 'GATE_LR_MULT', 1.0))
    lora_lr_mult = float(getattr(config, 'LORA_LR_MULT', 1.0))
    emb_lr_mult = float(getattr(config, 'EMB_LR_MULT', 1.0))

    lora_params_list = list(model.lora_adapters.parameters())
    gate_params_list = list(model.gates.parameters())
    emb_params_list = list(model.code_embeddings.parameters())
    encdec_params_list = (list(model.encoder.parameters())
                          + list(model.decoder.parameters()))

    lora_ids = {id(p) for p in lora_params_list}
    gate_ids = {id(p) for p in gate_params_list}
    emb_ids = {id(p) for p in emb_params_list}
    encdec_ids = {id(p) for p in encdec_params_list}
    grouped_ids = lora_ids | gate_ids | emb_ids | encdec_ids

    other_params = [p for p in model.parameters()
                    if p.requires_grad and id(p) not in grouped_ids]
    encdec_trainable = [p for p in encdec_params_list if p.requires_grad]

    emb_wd_eff = (config.EMB_WD if config.EMB_WD is not None
                  else config.WEIGHT_DECAY)
    gate_wd_eff = (config.GATE_WD if config.GATE_WD is not None
                   else config.WEIGHT_DECAY)
    enc_wd_eff = (getattr(config, 'ENCODER_WD', None)
                  if getattr(config, 'ENCODER_WD', None) is not None
                  else config.WEIGHT_DECAY)

    lora_lr_eff = config.LEARNING_RATE * lora_lr_mult
    gate_lr_eff = config.LEARNING_RATE * gate_lr_mult
    emb_lr_eff = config.LEARNING_RATE * emb_lr_mult
    enc_lr_eff = config.LEARNING_RATE * enc_lr_scale

    param_groups = []
    if lora_params_list:
        param_groups.append(
            {'params': lora_params_list,
             'lr': lora_lr_eff,
             'weight_decay': config.LORA_WD})
    if gate_params_list:
        param_groups.append(
            {'params': gate_params_list,
             'lr': gate_lr_eff,
             'weight_decay': gate_wd_eff})
    if encdec_trainable:
        param_groups.append(
            {'params': encdec_trainable,
             'lr': enc_lr_eff,
             'weight_decay': enc_wd_eff})
    if emb_params_list:
        param_groups.append(
            {'params': emb_params_list,
             'lr': emb_lr_eff,
             'weight_decay': emb_wd_eff})
    param_groups.append(
        {'params': other_params,
         'lr': config.LEARNING_RATE,
         'weight_decay': config.WEIGHT_DECAY})
    optimizer = torch.optim.Adam(param_groups)

    enc_msg = (f", enc_lr={enc_lr_eff:.2g} enc_wd={enc_wd_eff}"
               if encdec_trainable else "")
    print(f"Optimizer: Adam lr={config.LEARNING_RATE}{enc_msg} | "
          f"lora_lr={lora_lr_eff:.2g} lora_wd={config.LORA_WD} | "
          f"gate_lr={gate_lr_eff:.2g} gate_wd={gate_wd_eff} | "
          f"emb_lr={emb_lr_eff:.2g} emb_wd={emb_wd_eff} | "
          f"wd={config.WEIGHT_DECAY}")
    logging.info(
        f"Optimizer: Adam lr={config.LEARNING_RATE}{enc_msg} | "
        f"lora_lr={lora_lr_eff:.2g} lora_wd={config.LORA_WD} | "
        f"gate_lr={gate_lr_eff:.2g} gate_wd={gate_wd_eff} | "
        f"emb_lr={emb_lr_eff:.2g} emb_wd={emb_wd_eff} | "
        f"wd={config.WEIGHT_DECAY}")

    best_auc = float('-inf')
    best_logloss = float('inf')
    patience_counter = 0

    BEST_MODEL_PATH = f"{exp_dir}/best_model.pth"
    BEST_CODEBOOK_DIR = f"{exp_dir}/codebooks"
    os.makedirs(BEST_CODEBOOK_DIR, exist_ok=True)

    print("\nStarting training...")
    logging.info("Starting training...")
    for epoch in range(config.NUM_EPOCHS):
        tau_d = anneal_temperature(
            epoch, config.TAU_DEPTH_INIT, config.TAU_MIN, config.TAU_DECAY)
        model.set_temperature(tau_d)

        avgs, epoch_time = train_epoch(
            model, train_loader, optimizer, device, epoch, config)

        log_msg = (f"Epoch {epoch}: Loss {avgs['loss']:.4f} "
                   f"(CTR={avgs['ctr']:.4f} Dep={avgs['depth']:.2f} "
                   f"Anc={avgs['anchor']:.4f}) "
                   f"tau_d={tau_d:.3f} "
                   f"Time={epoch_time:.1f}s")
        print(log_msg)
        logging.info(log_msg)

        val_start = time.time()
        val_metrics = evaluate_model(model, val_loader, device)
        val_time = time.time() - val_start

        val_msg = (f"  Val AUC: {val_metrics['auc']:.5f}, "
                   f"LogLoss: {val_metrics['logloss']:.5f}, "
                   f"Time: {val_time:.2f}s")
        print(val_msg)
        logging.info(val_msg)

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_logloss = val_metrics['logloss']
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            for i in range(max_layers):
                cb = getattr(model, f'codebook_{i}').detach().cpu()
                with torch.no_grad():
                    delta = model._layer_delta(i).detach().cpu()
                artifact = {
                    'frozen': cb,
                    'delta': delta,
                    'lora_W_down':
                        model.lora_adapters[i].W_down.detach().cpu(),
                    'lora_W_up':
                        model.lora_adapters[i].W_up.detach().cpu(),
                    'emb_weight':
                        model.code_embeddings[i].weight.detach().cpu(),
                }
                torch.save(artifact,
                           os.path.join(BEST_CODEBOOK_DIR,
                                        f"layer{i+1}.pt"))
            save_msg = f"  Best model saved. AUC={best_auc:.5f}"
            print(save_msg)
            logging.info(save_msg)
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epoch(s)")
            logging.info(f"  No improvement for {patience_counter} epoch(s)")
            if patience_counter >= config.PATIENCE:
                stop_msg = f"Early stopping at epoch {epoch}"
                print(stop_msg)
                logging.info(stop_msg)
                break

    print("\nLoading best model for test...")
    state = torch.load(BEST_MODEL_PATH, map_location=device)
    incompat = model.load_state_dict(state, strict=False)
    if incompat.missing_keys or incompat.unexpected_keys:
        print(f"  [load_state_dict] missing={incompat.missing_keys}, "
              f"unexpected={incompat.unexpected_keys}")
        logging.info(
            f"load_state_dict missing={incompat.missing_keys}, "
            f"unexpected={incompat.unexpected_keys}")
    model.to(device)
    model.eval()

    test_start = time.time()
    test_metrics = evaluate_model(model, test_loader, device)
    test_time = time.time() - test_start

    test_msg = (f"Test AUC: {test_metrics['auc']:.5f}, "
                f"LogLoss: {test_metrics['logloss']:.5f}, "
                f"Time: {test_time:.2f}s")
    print(test_msg)
    logging.info(test_msg)
    print(f"best_val_auc {best_auc:.5f} best_val_logloss {best_logloss:.5f}")
    print(f"test_auc {test_metrics['auc']:.5f} "
          f"test_logloss {test_metrics['logloss']:.5f}")

    result = {
        'val_auc': float(best_auc),
        'val_logloss': float(best_logloss),
        'test_auc': float(test_metrics['auc']),
        'test_logloss': float(test_metrics['logloss']),
    }
    logging.shutdown()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    config = get_optst_config()
    run_training(config)


if __name__ == "__main__":
    main()
