import torch
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss


def calculate_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict:
    predictions = np.clip(predictions, 1e-7, 1 - 1e-7)

    auc = roc_auc_score(labels, predictions)
    logloss = log_loss(labels, predictions)

    return {
        'auc': auc,
        'logloss': logloss
    }


def evaluate_model(model, data_loader, device) -> dict:
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            profile_vectors = batch['profile_vector'].to(device)
            item_ids = batch['item_id'].to(device)
            user_ids = batch['user_id'].to(device)
            labels = batch['label'].to(device)

            features = {'user_id': user_ids, 'item_id': item_ids}

            predictions = model(profile_vectors, features)

            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0).flatten()
    all_labels = np.concatenate(all_labels, axis=0).flatten()

    metrics = calculate_metrics(all_predictions, all_labels)

    return metrics
