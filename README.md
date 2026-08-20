# OptST

Anonymous implementation accompanying the WSDM 2027 submission
"OptST: Optimizing Semantic Tokenization for LLM-Enhanced Recommendation."

This repository contains the core OptST pipeline with a DCN downstream
recommender: dataset preprocessing, training-only LLM profile generation,
RQ-VAE pretraining, and downstream task-aware semantic-token optimization.

## Environment

The code targets Python 3.10. Install the required packages with:

```bash
pip install -r requirements.txt
```

Profile generation requires a local Qwen3-8B checkpoint. By default, the code
expects it at `models/Qwen3-8B/`. The frozen profile encoder
`sentence-transformers/all-MiniLM-L6-v2` is loaded through
SentenceTransformers.

## Data layout

The public datasets and released split indices are not redistributed. Place
them under the following layout:

```text
CIKM20-NETE-Datasets/
├── Amazon/
│   ├── MoviesAndTV/
│   │   ├── reviews.pickle
│   │   ├── item.json
│   │   └── 3/{train,validation,test}.index
│   ├── Games/
│   │   └── ...
│   └── Clothing/
│       └── ...
└── Yelp/
    ├── reviews.pickle
    ├── item.json
    ├── user.json
    └── 3/{train,validation,test}.index
```

The profile-generation step requires `train.index`; review histories are
restricted to interactions in that partition. The script stops with an error if
the training split is unavailable.

## Run

The following example shows the complete Clothing workflow. The preprocessing
and profile-generation commands also accept `Movies`, `Yelp`, and `Games` via
the `--dataset` argument.

### 1. Prepare the CTR data and training-only profiles

Run these commands from `data_processing/`:

```bash
cd data_processing

python preprocess_csv.py --dataset Clothing
python local_llm_single_qwen.py --dataset Clothing
```

The first command converts observed ratings into binary labels using
`rating > 3`. No unobserved user-item pairs are sampled as additional
negatives. The second command summarizes the user's training-partition review
history with Qwen3-8B and encodes the resulting profile with frozen MiniLM.
Amazon prompts also use associated item metadata. Yelp prompts use item-category
metadata and the static activity fields available in `user.json`.

### 2. Pretrain the residual semantic tokenizer

Run from `OptST/`:

```bash
cd ../OptST

python pretrain_codebook.py --dataset Clothing \
    --max_layers 8 --codebook_size 256 --latent_dim 16 \
    --encoder_hidden 128 --beta 0.1 --rq_epochs 100 --rq_lr 3e-4 \
    --output_dir ./pretrained/Clothing/K256_L8_ld16_eh128_beta0.1_lr0.0003
```

### 3. Train OptST

```bash
python train.py --dataset Clothing \
    --pretrained_dir ./pretrained/Clothing/K256_L8_ld16_eh128_beta0.1_lr0.0003 \
    --learning_rate 5e-4 --weight_decay 1e-4 \
    --lora_rank 2 --lora_init_scale 0.003 --lora_lr_mult 1.0 --lora_wd 1e-4 \
    --depth_lambda 0.01 --anchor_lambda 0.0 \
    --gate_init_bias 1.0 --gate_lr_mult 1.0 \
    --no-finetune_encoder \
    --ste_temperature 1.0 \
    --tau_depth_init 1.0 --tau_min 0.1 --tau_decay 0.95 \
    --batch_size 2048 --num_epochs 100 --patience 5
```

The explicit `--no-finetune_encoder` option freezes the pretrained RQ encoder
during downstream training, matching the configuration described in the WSDM
submission. Model selection and early stopping use validation AUC. The selected
checkpoint is evaluated on the test partition only after training.

The default random seed is `3407`; pass `--seed` to run another seed.

## Outputs

- RQ-VAE checkpoints and metadata are written to the supplied `--output_dir`.
- OptST logs, the best validation checkpoint, and adapted-codebook artifacts are
  written under `OptST/exp/` by default.

Datasets, generated profiles, local LLM checkpoints, experiment logs, and model
checkpoints are excluded from version control.
